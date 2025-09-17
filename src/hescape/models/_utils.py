import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from hescape._utils import find_project_root

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


def gather_features(
    image_features,
    text_features,
    source_labels,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    assert has_distributed, "torch.distributed did not import correctly, please use a PyTorch version with support."

    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        all_labels = torch.cat(torch.distributed.nn.all_gather(source_labels), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        gathered_source_labels = [torch.zeros_like(source_labels) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gathered_source_labels, source_labels)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
            gathered_source_labels[rank] = source_labels
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        all_source_labels = torch.cat(gathered_source_labels, dim=0)

    return all_image_features, all_text_features, all_source_labels


class SourceAwareContrastiveLoss(nn.Module):
    """Source basedSigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_logits(self, image_features, text_features, source_labels, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features, all_source_labels = gather_features(
                image_features,
                text_features,
                source_labels,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                source_labels_mask = source_labels.unsqueeze(1) == source_labels.unsqueeze(0)
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                source_labels_mask = all_source_labels.unsqueeze(1) == all_source_labels.unsqueeze(0)
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            source_labels_mask = source_labels.unsqueeze(1) == source_labels.unsqueeze(0)

        return logits_per_image, source_labels_mask

    def forward(self, image_features, text_features, logit_scale, source_labels, output_dict=False):
        logits_per_image, source_labels_mask = self.get_logits(
            image_features, text_features, source_labels, logit_scale
        )

        # Convert boolean mask to float, excluding diagonal if you don't want self-positives
        # organ_mask_excl_diag = organ_mask & (~torch.eye(batch_size, dtype=bool, device=device))
        N = source_labels_mask.shape[0]
        device = source_labels_mask.device
        source_labels_mask = source_labels_mask & ~torch.eye(N, dtype=torch.bool, device=device)

        # For each row, sum or average log_probs of positives
        positives_per_row = source_labels_mask.sum(dim=1)  # how many positives per row

        # For stability, we compute the log-softmax row by row.
        log_probs = F.log_softmax(logits_per_image, dim=1)  # shape (batch_size, batch_size)

        eps = 1e-8
        source_aware_loss = 0.0

        for i in range(N):
            count_pos = positives_per_row[i]
            if count_pos > 0:
                row_pos_log_probs = log_probs[i][source_labels_mask[i]]
                # Typically we sum, then average by number_of_positives
                source_aware_loss -= row_pos_log_probs.mean()
            else:
                # If no positives, skip
                pass

        source_aware_loss /= N

        return {"contrastive_loss": loss} if output_dict else source_aware_loss


def print_trainable_parameters(name, model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"{name} trainable params: {trainable_params / 1e6:.2f}M || all params: {all_param / 1e6:.2f}M || trainable%: {100 * trainable_params / all_param}"
    )


def _crawl_and_import_custom_model():
    """Finds and dynamically imports the user's custom model definition."""
    # 1. Locate the project root (hescape/) by navigating up from this file.
    # This path is .../src/hescape/models/image_models/custom_encoder.py
    # We need to go up 4 levels to get to the root `hescape/` directory.
    project_root = Path(find_project_root())
    custom_model_dir = project_root / "custom_model"

    if not custom_model_dir.exists():
        raise FileNotFoundError(
            f"The 'custom_model' directory was not found at the project root. Expected location: {custom_model_dir}"
        )

    # 2. Crawl for .py files and import them
    py_files = list(custom_model_dir.rglob("*.py"))
    if not py_files:
        raise ImportError(f"No Python files found in {custom_model_dir} to import.")

    for path in py_files:
        # Create a unique module name to avoid conflicts
        module_name = f"custom_model.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"Dynamically imported user model from: {path.name}")

    return custom_model_dir

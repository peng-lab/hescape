import torch.nn as nn
from conch.open_clip_custom import create_model_from_pretrained


def _build_conch_model(path: str):
    return Conch(path)


class Conch(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        model, _ = create_model_from_pretrained("conch_ViT-B-16", str(path))

        self.num_features = model.embed_dim
        self.visual = model.visual
        self.visual.attn_pool_caption = nn.Identity()
        self.visual.ln_caption = nn.Identity()

    def forward(self, x):
        return self.visual.forward_no_head(x, normalize=False)

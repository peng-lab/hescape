import torch

from hescape.models.gexp_models.scf_utils import gatherData, load_model_frommmf


def _build_scfoundation_model(path: str):
    return scFoundationModel(path)


class scFoundationModel(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.ckpt_path = path
        self.build()

    def build(self):
        model, model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder

        self.norm = torch.nn.Identity()
        self.model_config = model_config

    def forward(self, x):
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config["pad_token_id"])
        data_gene_ids = torch.arange(19266, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.model_config["pad_token_id"])

        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x, x_padding)

        # mlp
        logits, _ = torch.max(logits, dim=1)  # b,dim

        logits = self.norm(logits)

        return logits

"""
Nicheformer: a foundation model for single-cell and spatial omics
© Alejandro Tejada-Lapuerta / Helmholtz Munich
"""

import torch
import torch.nn as nn

MASK_TOKEN = 0
CLS_TOKEN = 2


class Nicheformer(nn.Module):
    """The transformer-based foundation model Nicheformer."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 16,
        ffn_mult: int = 2,
        dropout: float = 0.0,
        batch_first: bool = True,
        n_layers: int = 12,
        n_tokens: int = 20340,
        context_length: int = 1500,
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            layer_norm_eps=1e-12,
            batch_first=batch_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # as in HuggingFace
        self.classifier_head = nn.Linear(d_model, n_tokens, bias=False)
        bias = nn.Parameter(torch.zeros(n_tokens))
        self.classifier_head.bias = bias

        # as in HuggingFace
        self.pooler_head = nn.Linear(d_model, d_model, bias=True)
        self.activation = nn.Tanh()
        self.cls_head = nn.Linear(d_model, 164, bias=True)

        # token embedding learnable weights
        self.embeddings = nn.Embedding(n_tokens + 5, d_model, padding_idx=1)

        # uses learnable weights as positional embeddings
        self.positional_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos = torch.arange(0, context_length, dtype=torch.long)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        token_embedding = self.embeddings(x)
        pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device))
        embeddings = self.dropout(token_embedding + pos_embedding)

        transformer_output = self.encoder(embeddings, src_key_padding_mask=attention_mask, is_causal=False)
        mlm_prediction = self.classifier_head(transformer_output)

        return {"mlm_prediction": mlm_prediction, "transformer_output": transformer_output}


if __name__ == "__main__":
    # Instantiate the model
    nicheformer = Nicheformer()

    # Path to the state dictionary
    PATH = "./model_files/nicheformer.ckpt"

    # Load the state dictionary
    nicheformer.load_state_dict(torch.load(PATH)["state_dict"], strict=False)

    # Test Nicheformer forward pass
    nicheformer = nicheformer.cuda()
    gene_tokens = torch.randint(0, 9, (1, 1500)).cuda()
    output = nicheformer(gene_tokens, None)
    print(output["mlm_prediction"].shape, output["transformer_output"].shape)
    print("Nicheformer output:", len(output))

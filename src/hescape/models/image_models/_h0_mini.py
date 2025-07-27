import timm
import torch
import torch.nn as nn


def _build_h0_mini_model(path: str):
    return H0_mini(path)


class H0_mini(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        model = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        self.trunk = model
        self.trunk.load_state_dict(torch.load(path, weights_only=True), strict=True)

    def forward(self, x):
        return self.trunk(x)

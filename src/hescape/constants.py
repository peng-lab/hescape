from __future__ import annotations

from enum import Enum

import torch
import torchvision.transforms.v2 as T


class DatasetEnum(str, Enum):
    """Enum of datasets covariates."""

    NAME = "name"
    IMG = "image"
    GEXP = "gexp"
    COORDS = "cell_coords"
    SOURCE = "source"
    ATLAS = "atlas"
    AGE = "age"
    DIAGNOSIS = "diagnosis"
    CANCER = "cancer"
    ONCOTREE_CODE = "oncotree_code"
    TISSUE = "tissue"
    TUMOR_GRADE = "tumor_grade"
    GENDER = "gender"
    RACE = "race"
    TREATMENT_TYPE = "treatment_type"
    THERAPEUTIC_AGENTS = "therapeutic_agents"
    TUMOR_TISSUE_TYPE = "tumor_tissue_type"
    ASSAY = "assay"
    PRESERVATION_METHOD = "preservation_method"
    STAIN = "stain"
    SPACERANGER = "spaceranger"
    SPECIES = "species"
    CYTASSIST = "cytassist"


EVAL_TRANSFORMS = {
    "conch": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512)]),
            T.Resize((480, 480), antialias=True, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((480, 480)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    ),
    "optimus": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ]
    ),
    "h0-mini": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ]
    ),
    "uni": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "ctranspath": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
    "gigapath": T.Compose(
        [
            T.ToImage(),
            T.RandomChoice([T.CenterCrop(512), T.CenterCrop(256)]),
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
}

# fix some parameters.
# We always need to apply the crop transform, but before this can be applied,
# we need to convert PIL.Image to Tensor.


import torch
from torchvision.transforms import v2 as transf_v2

import data_setup

ANNOTATION_LEVEL = "family"
DATA_ROOT_DIR = "data"
DATA_AUGMENTATION_TRANSFORMS = transf_v2.TrivialAugmentWide()
CROP_TRANSFORM = transf_v2.Compose(
    [
        transf_v2.ToImage(),
        transf_v2.ToDtype(torch.float32, scale=True),
        data_setup.CropAuthorshipInformation(),
    ]
)
BATCH_SIZE = 32

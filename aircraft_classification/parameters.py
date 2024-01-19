# fix some parameters.
# We always need to apply the crop transform, but before this can be applied,
# we need to convert PIL.Image to Tensor.


import torch
from torchvision.transforms import v2 as transf_v2


ANNOTATION_LEVEL = "family"
DATA_ROOT_DIR = "data"
DATA_AUGMENTATION_TRANSFORMS = transf_v2.TrivialAugmentWide()


BATCH_SIZE = 32

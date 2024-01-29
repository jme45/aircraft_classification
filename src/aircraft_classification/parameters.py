# fix some parameters.
# We always need to apply the crop transform, but before this can be applied,
# we need to convert PIL.Image to Tensor.


from pathlib import Path
from torchvision.transforms import v2 as transf_v2

# Get the root dir of the package, so we can put the data into <Root>/data
# Need to take .parent 3 times, since taking it once only gets you to the
# directory the file is in and we need to go up 2 directories.
root_dir = Path(__file__).parent.parent.parent


ANNOTATION_LEVEL = "family"
DATA_ROOT_DIR = root_dir / "data"
DEFAULT_RUNS_DIR = root_dir / "runs"
DATA_AUGMENTATION_TRANSFORMS = transf_v2.TrivialAugmentWide()


BATCH_SIZE = 32

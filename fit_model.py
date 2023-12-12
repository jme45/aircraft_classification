import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Any
import pandas as pd

import aircraft_classification
import data_setup
import aircraft_types
import ml_utils
from torchvision.transforms import v2 as transf_v2
from torch.utils.data import DataLoader
import torch
from torch import nn

# In colab and locally it seems that ml_utils gets installed differently.
# In case ClassificationTrainer is not in ml_utils, import ml_utils.ml_utils
# if "ClassificationTrainer" not in dir(ml_utils):
#     from ml_utils import ml_utils

# fix some parameters.
ANNOTATION_LEVEL = "family"
DATA_ROOT_DIR = "data"
DATA_AUGMENTATION_TRANSFORMS = transf_v2.TrivialAugmentWide()
# We always need to apply the crop transform, but before this can be applied,
# we need to convert PIL.Image to Tensor.
CROP_TRANSFORM = transf_v2.Compose(
    [
        transf_v2.ToImage(),
        transf_v2.ToDtype(torch.float32, scale=True),
        data_setup.CropAuthorshipInformation(),
    ]
)
BATCH_SIZE = 32


def fit_model(
    model_type: str,
    aircraft_subset_name: str,
    n_epochs: int,
    output_path: str | Path = Path("runs"),
    compile_model: bool = False,
    device="cuda",
    num_workers: int = 0,
    experiment_name: str = "test",
) -> Tuple[dict[str, list], dict[str, Any]]:
    device = device if torch.cuda.is_available() else "cpu"
    # More workers are only useful if using CUDA (experimentally).
    # I won't ever have access to a Computer with more than one GPU,
    # so can cap number of workers at 2. Similarly pin_memory.
    if device == "cuda":
        num_workers = min(num_workers, 2)
        num_workers = min(num_workers, os.cpu_count())
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    # Set up output_path, consisting of experiment_name, etc.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%m")
    output_path = (
        Path(output_path)
        / experiment_name
        / aircraft_subset_name
        / model_type
        / timestamp
    )

    class_names = aircraft_types.AIRCRAFT_SUBSETS[aircraft_subset_name.upper()]

    classifier = aircraft_classification.AircraftClassifier(
        model_type, class_names, load_classifier_pretrained_weights=False
    )
    if compile_model:
        classifier.model = torch.compile(classifier.model)

    # Calculate number of model parameters
    num_params = sum(torch.numel(param) for param in classifier.model.parameters())

    # Train transform consists of data augmentation transform + model transforms.
    # Validation transform consists only of model transforms.
    train_transforms = transf_v2.Compose(
        [CROP_TRANSFORM, DATA_AUGMENTATION_TRANSFORMS, classifier.transforms]
    )
    # For validation, don't use data augmentation
    val_transforms = transf_v2.Compose([CROP_TRANSFORM, classifier.transforms])

    train_set = data_setup.get_aircraft_data_subset(
        "data",
        "train",
        ANNOTATION_LEVEL,
        train_transforms,
        target_transform=None,
        download=True,
        aircraft_subset_name=aircraft_subset_name,
    )
    val_set = data_setup.get_aircraft_data_subset(
        "data",
        "val",
        ANNOTATION_LEVEL,
        val_transforms,
        target_transform=None,
        download=True,
        aircraft_subset_name=aircraft_subset_name,
    )

    train_dataloader = DataLoader(
        train_set,
        BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_set,
        BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    tensorboard_logger = ml_utils.TensorBoardLogger(True, root_dir=output_path)
    trainer = ml_utils.ClassificationTrainer(
        model=classifier.model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimiser_class="Adam",
        optimiser_kwargs={"lr": 1e-3},
        loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1),
        n_epochs=n_epochs,
        device=device,
        output_path=output_path,
        num_classes=len(class_names),
        save_lowest_test_loss_model=True,
        save_final_model=True,
        tensorboard_logger=tensorboard_logger,
        disable_epoch_progress_bar=False,
        disable_within_epoch_progress_bar=False,
    )

    all_results = trainer.train()

    meta_info = dict(num_params=num_params, output_path=str(output_path))

    return all_results, meta_info


if __name__ == "__main__":
    all_results, meta_info = fit_model("trivial", "TEST", 2)

    print("For run:")
    print(meta_info)
    print("\nOutput frame:")
    print(pd.DataFrame(all_results))

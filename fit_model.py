import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Any
import pandas as pd

import classifiers
import data_setup
import aircraft_types
from ml_utils import ml_utils
from torchvision.transforms import v2 as transf_v2
from torch.utils.data import DataLoader
import torch
from torch import nn

import parameters

# In colab and locally it seems that ml_utils gets installed differently.
# In case ClassificationTrainer is not in ml_utils, import ml_utils_colab.ml_utils.
# This is not needed, due to the way I set up the colab notebook.
# if "ClassificationTrainer" not in dir(ml_utils):
#     from ml_utils_colab.ml_utils import ml_utils


def fit_model(
    model_type: str,
    aircraft_subset_name: str,
    n_epochs: int,
    output_path: str | Path = Path("runs"),
    compile_model: bool = False,
    device="cuda",
    num_workers: int = 0,
    experiment_name: str = "test",
    print_progress_to_screen: bool = False,
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = (
        Path(output_path)
        / experiment_name
        / aircraft_subset_name
        / model_type
        / timestamp
    )

    classifier = classifiers.AircraftClassifier(
        model_type, aircraft_subset_name, load_classifier_pretrained_weights=False
    )
    if compile_model:
        classifier.model = torch.compile(classifier.model)

    # Calculate number of model parameters
    num_params = sum(torch.numel(param) for param in classifier.model.parameters())

    # Train transform consists of data augmentation transform + model transforms.
    # Validation transform consists only of model transforms.
    train_transforms = transf_v2.Compose(
        [
            parameters.TO_TENSOR_TRANSFORMS,
            data_setup.CropAuthorshipInformation(),
            parameters.DATA_AUGMENTATION_TRANSFORMS,
            classifier.transforms,
        ]
    )
    # For validation, don't use data augmentation
    val_transforms = transf_v2.Compose(
        [
            parameters.TO_TENSOR_TRANSFORMS,
            parameters.CROP_TRANSFORM,
            classifier.transforms,
        ]
    )

    train_set = data_setup.get_aircraft_data_subset(
        "data",
        "train",
        parameters.ANNOTATION_LEVEL,
        train_transforms,
        target_transform=None,
        download=True,
        aircraft_subset_name=aircraft_subset_name,
    )
    val_set = data_setup.get_aircraft_data_subset(
        "data",
        "val",
        parameters.ANNOTATION_LEVEL,
        val_transforms,
        target_transform=None,
        download=True,
        aircraft_subset_name=aircraft_subset_name,
    )

    train_dataloader = DataLoader(
        train_set,
        parameters.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_set,
        parameters.BATCH_SIZE,
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
        num_classes=classifier.num_classes,
        save_lowest_test_loss_model=True,
        save_final_model=True,
        tensorboard_logger=tensorboard_logger,
        disable_epoch_progress_bar=False,
        disable_within_epoch_progress_bar=False,
        print_progress_to_screen=print_progress_to_screen,
        state_dict_extractor=classifier.state_dict_extractor,
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

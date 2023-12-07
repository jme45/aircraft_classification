import os
from pathlib import Path

import aircraft_classification
import data_setup
import aircraft_types
import ml_utils
from torchvision.transforms import v2 as transf_v2
from torch.utils.data import DataLoader
import torch
from torch import nn

# fix some parameters.
ANNOTATION_LEVEL = "family"
DATA_ROOT_DIR = "data"
DATA_AUGMENTATION_TRANSFORMS = transf_v2.TrivialAugmentWide()
PIN_MEMORY = True
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32


def fit_model(
    model_type: str,
    aircraft_types_description: str,
    n_epochs: int,
    output_path: str | Path = Path("runs"),
    compile_model: bool = False,
    device="cuda",
):
    device = device if torch.cuda.is_available() else "cpu"

    root = Path(output_path) / aircraft_types_description / model_type

    class_names = getattr(aircraft_types, aircraft_types_description.upper())

    classifier = aircraft_classification.AircraftClassifier(
        model_type, class_names, load_classifier_pretrained_weights=False
    )

    # Train transform consists of data augmtation transform + model transforms.
    # Validation transform consists only of model transforms.
    train_transform = transf_v2.Compose(
        [DATA_AUGMENTATION_TRANSFORMS, classifier.transforms]
    )

    train_set = data_setup.get_aircraft_data_subset(
        "data",
        "train",
        ANNOTATION_LEVEL,
        train_transform,
        target_transform=None,
        download=True,
        aircraft_types=class_names,
    )
    val_set = data_setup.get_aircraft_data_subset(
        "data",
        "val",
        ANNOTATION_LEVEL,
        classifier.transforms,
        target_transform=None,
        download=True,
        aircraft_types=class_names,
    )

    train_dataloader = DataLoader(
        train_set,
        BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_dataloader = DataLoader(
        val_set,
        BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    tensorboard_logger = ml_utils.TensorBoardLogger(True, "test", root_dir=output_path)
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

    trainer.train()


if __name__ == "__main__":
    fit_model("trivial", "TEST", 2)

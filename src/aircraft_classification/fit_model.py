"""
Module to run a model fit without too much effort.
"""


import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any

import pandas as pd
import torch
from ml_utils_jme45 import ml_utils
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from aircraft_classifiers_jme45 import classifiers
from . import data_setup
from . import parameters


# In colab and locally it seems that ml_utils gets installed differently.
# In case ClassificationTrainer is not in ml_utils, import ml_utils_colab.ml_utils.
# This is not needed, due to the way I set up the colab notebook.
# if "ClassificationTrainer" not in dir(ml_utils):
#     from ml_utils_colab.ml_utils import ml_utils


def fit_model(
    model_type: str,
    aircraft_subset_name: str,
    n_epochs: int,
    output_path: str | Path = parameters.DEFAULT_RUNS_DIR,
    compile_model: bool = False,
    device="cpu",
    num_workers: int = 0,
    experiment_name: str = "test",
    print_progress_to_screen: bool = False,
    optimiser_class: str | Optimizer = "Adam",
    optimiser_kwargs: dict = {"lr": 1e-3},
    return_classifier: bool = False,
) -> Tuple[dict[str, list], dict[str, Any]]:
    """
    Fit a model of a particular type to an aircraft subset

    :param model_type: type of model, e.g. vit_b_16, effnet_b7
    :param aircraft_subset_name: e.g. CIVILIAN_JETS
    :param n_epochs: number of epochs to train for
    :param output_path: path to save state dict and tensorboard files
    :param compile_model: whether to compile (apparently best on good GPUs)
    :param device: device to run on.
    :param num_workers: num workers for dataloader. 0 best on laptop.
    :param experiment_name:
    :param print_progress_to_screen:
    :param optimiser_class: e.g. "Adam" or "SGD"
    :param optimiser_kwargs: any arguments for the optimiser, e.g. "lr"
    :param return_classifier: If True, return the final classifier as an extra element in the list returned.
    :return:
    """
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
        model_type,
        aircraft_subset_name,
        load_classifier_pretrained_weights=False,
        data_augmentation_transforms=parameters.DATA_AUGMENTATION_TRANSFORMS,
    )

    # Compiling is allegedly useful on more powerful GPUs.
    if compile_model:
        classifier.model = torch.compile(classifier.model)

    # Calculate number of model parameters, mainly as general info.
    num_params = sum(torch.numel(param) for param in classifier.model.parameters())

    # Set up training and validation sets.
    train_set = data_setup.get_aircraft_data_subset(
        root=parameters.DATA_ROOT_DIR,
        split="train",
        annotation_level=parameters.ANNOTATION_LEVEL,
        transform=classifier.train_transform_with_crop,
        target_transform=None,
        download=True,
        aircraft_subset_name=aircraft_subset_name,
    )
    val_set = data_setup.get_aircraft_data_subset(
        root=parameters.DATA_ROOT_DIR,
        split="val",
        annotation_level=parameters.ANNOTATION_LEVEL,
        transform=classifier.predict_transform_with_crop,
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

    # Set up tensorboard logging.
    tensorboard_logger = ml_utils.TensorBoardLogger(True, root_dir=output_path)

    # Define a trainer, which will do the training on the model.
    trainer = ml_utils.ClassificationTrainer(
        model=classifier.model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimiser_class=optimiser_class,
        optimiser_kwargs=optimiser_kwargs,
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
        trainable_parts=classifier.trainable_parts,
    )

    # Now run the training.
    all_results = trainer.train()

    # Obtain information about the model and training run.
    meta_info = dict(num_params=num_params, output_path=str(output_path))

    # List of items to return. If we want to also return the classifier, add it to the list.
    ret = [all_results, meta_info]
    if return_classifier:
        ret.append(classifier)

    return ret


if __name__ == "__main__":
    all_results, meta_info = fit_model("trivial", "TEST", 2)

    print("For run:")
    print(meta_info)
    print("\nOutput frame:")
    print(pd.DataFrame(all_results))

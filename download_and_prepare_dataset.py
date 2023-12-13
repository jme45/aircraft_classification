"""
Download dataset once, so it is available
"""

import data_setup
import fit_model


def run_downloads():
    train_set = data_setup.AircraftData(
        fit_model.DATA_ROOT_DIR,
        "train",
        fit_model.ANNOTATION_LEVEL,
        transform=data_setup.simple_transf,
        download=True,
    )
    val_set = data_setup.AircraftData(
        fit_model.DATA_ROOT_DIR,
        "val",
        fit_model.ANNOTATION_LEVEL,
        transform=data_setup.simple_transf,
        download=True,
    )


if __name__ == "__main__":
    run_downloads()

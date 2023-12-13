"""
Download dataset once, so it is available
"""

import data_setup
import parameters


def run_downloads():
    train_set = data_setup.AircraftData(
        parameters.DATA_ROOT_DIR,
        "train",
        parameters.ANNOTATION_LEVEL,
        transform=data_setup.simple_transf,
        download=True,
    )
    val_set = data_setup.AircraftData(
        parameters.DATA_ROOT_DIR,
        "val",
        parameters.ANNOTATION_LEVEL,
        transform=data_setup.simple_transf,
        download=True,
    )


if __name__ == "__main__":
    run_downloads()

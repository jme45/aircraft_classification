from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transf_v2
from tqdm.auto import tqdm

from . import aircraft_types as act


class CropAuthorshipInformation(torch.nn.Module):
    """
    The lowest 20 pixels contain the authorship information for the picture
    in the FGVCAircraft dataset. This needs to be removed for training and testing.
    See: https://arxiv.org/pdf/1306.5151.pdf , page 3

    This class crops those last 20 pixel rows. It only works on tensors.
    """

    n_annotation_pixels = 20

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        # from the last dimension, drop the last n_annotation_pixels rows.
        return x[..., : -self.n_annotation_pixels]


simple_transf = transf_v2.Compose(
    [
        transf_v2.ToImage(),
        transf_v2.ToDtype(torch.float32, scale=True),
        CropAuthorshipInformation(),
        transf_v2.Resize((224, 224), antialias=True),
    ]
)


class AircraftData(torchvision.datasets.FGVCAircraft):
    """
    Subclass of FGVCAircraft, which also stores all the targets in a list, so they are easily accessible.

    First time this is called, the list needs to be cached, which may take a minute or more.
    """

    # Using a dataloader to obtain targets can be a little faster.
    use_dataloader_to_find_targets = True
    BATCH_SIZE_FOR_TARGETS = 32

    def __init__(
        self,
        root: str | Path,
        split: str,
        annotation_level: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """

        :param root: root directory for storing of data
        :param split:  The dataset split, supports train, val, trainval and test.
        :param annotation_level: The annotation level, supports variant, family and manufacturer.
        :param transform: A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        :param target_transform: A function/transform that takes in the target and transforms it.
        :param download:  If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            str(root), split, annotation_level, transform, target_transform, download
        )

        self.targets_file = Path(root) / f"{split}_{annotation_level}_targets.txt"

        # If the file exists, open it and load the targets.
        if self.targets_file.exists():
            with self.targets_file.open("r") as f:
                # Targets are saved one target per line, so split by "\n" and apply int.
                self.targets = list(map(int, f.read().split("\n")))
        else:
            # Need to extract y values to get the targets.
            if self.use_dataloader_to_find_targets:
                # Make a temporary dataloader, to load X and y values of dataset.
                dl_tmp = DataLoader(self, self.BATCH_SIZE_FOR_TARGETS, shuffle=False)
                targets = []
                for X, y in tqdm(dl_tmp, desc="Extracting targets", total=len(dl_tmp)):
                    targets.append(y.numpy())
                targets = list(np.concatenate(targets))

            else:
                targets = []
                # Iterate through all items in dataset.
                for Xy in tqdm(self, desc="Extracting targets", total=len(self)):
                    targets.append(Xy[1])
            self.targets = targets

            # Write targets to file for quick retrieval next time.
            # Only write if no target transform.
            if target_transform is None:
                with self.targets_file.open("w") as f:
                    f.write("\n".join(map(str, self.targets)))

        # Sanity check.
        assert len(self.targets) == len(self)


def get_aircraft_data_subset(
    root: str | Path,
    split: str,
    annotation_level: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = True,
    aircraft_subset_name: str = "ALL_AIRCRAFT",
) -> AircraftData:
    """
    Load an AircraftData dataset and take the subset which only contain aircraft
    types given among aircraft_types.

    Labels (targets) are rescaled, so targets are 0..N-1 where N=len(aircraft_types)

    :param root: root directory for storing of data
    :param split:  The dataset split, supports train, val, trainval and test.
    :param annotation_level: The annotation level, supports variant, family and manufacturer.
    :param transform: A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
    :param target_transform: A function/transform that takes in the target and transforms it.
    :param download:  If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    :param aircraft_subset_name: Name of list of aircraft to keep.
        Defined in aircraft_types.py. E.g. "CIVILIAN_JET_AIRLINERS"
    :return: AircraftData containing only aircraft data from aircraft_types.
    """
    old_to_new_tgt_transform = act.TargetTransform(aircraft_subset_name)

    # Obtain new_target_transform. Need to compose if existing target_transform is not None.
    if target_transform is None:
        new_target_transform = transf_v2.Compose([old_to_new_tgt_transform])
    else:
        # Do old_to_new_tgt_transform first, since we first have to remove the gaps in the targets.
        new_target_transform = transf_v2.Compose(
            [
                old_to_new_tgt_transform,
                target_transform,
            ]
        )

    # Now that we know which target_transform needs to be applied, we can load AircraftData.
    # We cannot subsequently modify the transform, so we need to figure out the transform first.
    aircraft_data_all = AircraftData(
        root, split, annotation_level, transform, new_target_transform, download
    )
    # Now check that the aircraft types in aircraft_data_all matches ac.ALL_AIRCRAFT.
    # If not, the _map_old_classes_to_new will not work correctly.
    assert list(act.AIRCRAFT_SUBSETS["ALL_AIRCRAFT"]) == list(
        aircraft_data_all.classes
    ), "ALL_AIRCRAFT doesn't match classes in aircraft_data_all"

    # Get the indices of datapoints which are in the aircraft subset required.
    # The required indices are stored in old_to_new_tgt_transform
    idxs_subset = np.where(
        np.isin(
            aircraft_data_all.targets,
            old_to_new_tgt_transform.idxs_of_aircraft_in_subset,
        )
    )[0]

    # Save the targets. These need to be transformed later.
    untransformed_targets = np.array(aircraft_data_all.targets)[idxs_subset]

    # Now use torch to get a subset of the data.
    subset = torch.utils.data.Subset(aircraft_data_all, idxs_subset)

    # Need to apply transform to the untransformed targets.
    subset.targets = list(map(new_target_transform, untransformed_targets))
    # Since we now have fewer classes, we need to update the classes.
    subset.classes = aircraft_subset_name

    return subset

import torchvision

from pathlib import Path
from tqdm.auto import tqdm
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.transforms import v2 as transf_v2
from typing import Optional, Callable

import aircraft_types as act

simple_transf = transf_v2.Compose(
    [
        transf_v2.Resize((224, 224)),
        transf_v2.ToImage(),
        transf_v2.ToDtype(torch.float32, scale=True),
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
    aircraft_types: list[str] = act.ALL_AIRCRAFT,
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
    :param aircraft_types: List of aircraft to keep. E.g. ["A300", "Boeing 707"]
    :return: AircraftData containing only aircraft data from aircraft_types.
    """
    # sort aircraft_types and check that then get same order as in original classes
    aircraft_types = sorted(aircraft_types)
    assert set(aircraft_types).issubset(set(act.ALL_AIRCRAFT))
    aircraft_reduced = [a for a in act.ALL_AIRCRAFT if a in aircraft_types]
    assert list(aircraft_reduced) == list(
        aircraft_types
    ), "Order doesn't match. Should not happen"

    # Get the indices of the aircraft types among all the aircraft for which there is data.
    # Later check that all aircraft in AircraftData matches ac.ALL_AIRCRAFT.
    idxs_aircraft_types = [
        i
        for i, aircr_type in enumerate(act.ALL_AIRCRAFT)
        if aircr_type in aircraft_types
    ]

    # Modify target transforms. First get a dictionary from old target to new target.
    dict_old_new = {
        old_idx: new_idx for new_idx, old_idx in enumerate(idxs_aircraft_types)
    }
    # Anything that we don't want (aircraft not in aircraft_types) gets mapped to -1.
    _map_old_classes_to_new = lambda x: dict_old_new.get(x, -1)

    # Obtain new_target_transform. Need to compose if existing target_transform is not None.
    if target_transform is None:
        # FIXME: Something is wrong with transform. put proper transform back again transf_v2.Compose([transf_v2.Identity()])#
        new_target_transform = transf_v2.Compose([_map_old_classes_to_new])
    else:
        new_target_transform = transf_v2.Compose(
            [target_transform, _map_old_classes_to_new]
        )

    # Now that we know which target_transform needs to be applied, we can load AircraftData.
    # We cannot subsequently modify the transform, so we need to figure out the transform first.
    aircraft_data_all = AircraftData(
        root, split, annotation_level, transform, new_target_transform, download
    )
    # Now check that the aircraft types in aircraft_data_all matches ac.ALL_AIRCRAFT.
    # If not, the _map_old_classes_to_new will not work correctly.
    assert list(act.ALL_AIRCRAFT) == list(
        aircraft_data_all.classes
    ), "ac.ALL_AIRCRAFT doesn't match classes in aircraft_data_all"

    # Get the indices of datapoints which are in the aircraft subset required.
    idxs_subset = np.where(np.isin(aircraft_data_all.targets, idxs_aircraft_types))[0]

    # Save the targets. These need to be transformed later.
    untransformed_targets = np.array(aircraft_data_all.targets)[idxs_subset]

    # Now use torch to get a subset of the data.
    subset = torch.utils.data.Subset(aircraft_data_all, idxs_subset)

    # Need to apply transform to the untransformed targets.
    subset.targets = list(map(new_target_transform, untransformed_targets))
    # Since we now have fewer classes, we need to update the classes.
    subset.classes = aircraft_types

    return subset

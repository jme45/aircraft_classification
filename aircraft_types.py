AIRCRAFT_SUBSETS = {
    # Reduced set for testing purposes.
    "TEST": ["A380", "Boeing 747", "DC-8"],
    "CIVILIAN_JETS": [
        "A300",
        "A310",
        "A320",
        "A330",
        "A340",
        "A380",
        "Boeing 707",
        "Boeing 717",
        "Boeing 727",
        "Boeing 737",
        "Boeing 747",
        "Boeing 757",
        "Boeing 767",
        "Boeing 777",
        "DC-8",
        "DC-9",
        "DC-10",
        "MD-11",
        "MD-80",
        "MD-90",
        "Tu-134",
        "Tu-154",
    ],
    # ALL_AIRCRAFT is also a subset of ALL_AIRCRAFT, just not a proper subset
    "ALL_AIRCRAFT": [
        "A300",
        "A310",
        "A320",
        "A330",
        "A340",
        "A380",
        "ATR-42",
        "ATR-72",
        "An-12",
        "BAE 146",
        "BAE-125",
        "Beechcraft 1900",
        "Boeing 707",
        "Boeing 717",
        "Boeing 727",
        "Boeing 737",
        "Boeing 747",
        "Boeing 757",
        "Boeing 767",
        "Boeing 777",
        "C-130",
        "C-47",
        "CRJ-200",
        "CRJ-700",
        "Cessna 172",
        "Cessna 208",
        "Cessna Citation",
        "Challenger 600",
        "DC-10",
        "DC-3",
        "DC-6",
        "DC-8",
        "DC-9",
        "DH-82",
        "DHC-1",
        "DHC-6",
        "DR-400",
        "Dash 8",
        "Dornier 328",
        "EMB-120",
        "Embraer E-Jet",
        "Embraer ERJ 145",
        "Embraer Legacy 600",
        "Eurofighter Typhoon",
        "F-16",
        "F/A-18",
        "Falcon 2000",
        "Falcon 900",
        "Fokker 100",
        "Fokker 50",
        "Fokker 70",
        "Global Express",
        "Gulfstream",
        "Hawk T1",
        "Il-76",
        "King Air",
        "L-1011",
        "MD-11",
        "MD-80",
        "MD-90",
        "Metroliner",
        "PA-28",
        "SR-20",
        "Saab 2000",
        "Saab 340",
        "Spitfire",
        "Tornado",
        "Tu-134",
        "Tu-154",
        "Yak-42",
    ],
}


class TargetTransform:
    """
    Obtain target transform when using a subset of the available aicraft in the dataset.

    The targets (labels) in the FGVCAircraft are all the aircraft in ALL_AIRCRAFT.
    There are 70 aircraft, so the target will range from 0..69. If we only want to
    load a subset of aircraft, e.g. A300, Boeing 707 and Yak-42, we will only get the
    aircraft with targets 0, 12 and 69. The target transform transforms this so the three
    aircraft targets will be 0, 1, 2.

    """

    def __init__(self, aircraft_subset_name: str):
        assert (
            aircraft_subset_name.upper() in AIRCRAFT_SUBSETS
        ), f"aircraft_subset_name = {aircraft_subset_name} undefined."
        # sort aircraft_types and check that then get same order as in original classes
        aircraft_types = sorted(AIRCRAFT_SUBSETS[aircraft_subset_name.upper()])
        assert set(aircraft_types).issubset(set(AIRCRAFT_SUBSETS["ALL_AIRCRAFT"]))
        aircraft_reduced = [
            a for a in AIRCRAFT_SUBSETS["ALL_AIRCRAFT"] if a in aircraft_types
        ]
        assert list(aircraft_reduced) == list(
            aircraft_types
        ), "Order doesn't match. Should not happen"

        # Get the indices of the aircraft types among all the aircraft for which we want data.
        self.idxs_of_aircraft_in_subset = [
            i
            for i, aircr_type in enumerate(AIRCRAFT_SUBSETS["ALL_AIRCRAFT"])
            if aircr_type in aircraft_types
        ]
        # Modify target transforms. First get a dictionary from old target to new target.
        self.dict_old_new = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(self.idxs_of_aircraft_in_subset)
        }

    def __call__(self, old_target: int) -> int:
        """
        Get new target from old target. Return -1 there is no corresponding new target.
        :param old_target
        :return: new target
        """
        return self.dict_old_new.get(old_target, -1)

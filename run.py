"""
File for running the fit.
"""

import argparse

import pandas as pd

import aircraft_types
import fit_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="AircraftClassifierFitter.", description="Fits a classifier for aircraft."
    )

    parser.add_argument("-m", "--model-type")
    parser.add_argument(
        "-a",
        "--aircraft-subset-name",
        help=f"Name for aircraft subset. One of {list(aircraft_types.AIRCRAFT_SUBSETS.keys())} ",
    )
    parser.add_argument("-n", "--n-epochs", type=int, help="Number of epochs.")
    parser.add_argument(
        "-p", "--output-path", help="Location for output files", default="runs"
    )
    parser.add_argument(
        "-d", "--device", help="Device for calculation.", default="cuda"
    )
    parser.add_argument("--num-workers", help="Num workers for DataLoader.", default=0)
    parser.add_argument(
        "-e", "--experiment-name", help="Name for experiment.", default="test"
    )

    args = parser.parse_args()

    # Run the code
    all_results, meta_info = fit_model.fit_model(
        model_type=args.model_type,
        aircraft_subset_name=args.aircraft_subset_name,
        n_epochs=args.n_epochs,
        output_path=args.output_path,
        device=args.device,
        num_workers=args.num_workers,
        experiment_name=args.experiment_name,
    )

    print("For run:")
    print(meta_info)
    print("\nOutput frame:")
    print(pd.DataFrame(all_results))

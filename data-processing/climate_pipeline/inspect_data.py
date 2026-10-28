
from pathlib import Path
import xarray as xr

# Locate the repository root from this script's location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "rainfall"

# Find NetCDF files placed in the rainfall folder
files = list(DATA_DIR.glob("*.nc")) + list(DATA_DIR.glob("*.nc4"))

if not files:
    print("No NetCDF files found in:", DATA_DIR)
    print("Place an IMD rainfall .nc or .nc4 file in this folder first.")
else:
    for file_path in files:
        print(f"\nInspecting: {file_path.name}")

        try:
            with xr.open_dataset(file_path) as dataset:
                print("\nDataset summary:")
                print(dataset)

                print("\nDimensions:")
                print(dict(dataset.sizes))

                print("\nCoordinates:")
                for name, coord in dataset.coords.items():
                    print(
                        name,
                        "| dimensions:", coord.dims,
                        "| units:", coord.attrs.get("units", "not specified")
                    )

                print("\nData variables:")
                for name, variable in dataset.data_vars.items():
                    print(
                        name,
                        "| dimensions:", variable.dims,
                        "| units:", variable.attrs.get("units", "not specified"),
                        "| missing values:", int(variable.isnull().sum().compute().item())
                    )

        except Exception as error:
            print("Could not inspect this file:", error)

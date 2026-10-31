import json
from pathlib import Path

manifest_path = Path("data-processing/climate_pipeline/manifest.json")

with manifest_path.open(encoding="utf-8") as file:
    manifest = json.load(file)

print("Manifest JSON: valid")

for output in manifest["outputs"]:
    file_path = Path(output["file"])
    status = "exists" if file_path.exists() else "MISSING"
    print(f'{output["variable"]}: {status}')
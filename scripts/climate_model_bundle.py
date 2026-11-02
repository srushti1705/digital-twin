"""Package or install the trained climate model artifacts outside Git."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import joblib

from backend import climate_forecast

VARIABLES = ("rainfall", "tmax", "tmin")
MANIFEST_NAME = "manifest.json"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
USER_AGENT = "IndiaClimateTwin/1.0 (model artifact bootstrap)"


def _archive_names() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for variable in VARIABLES:
        files[f"models/climate_forecasts/{variable}.joblib"] = (
            climate_forecast.MODEL_DIRECTORY / f"{variable}.joblib"
        )
        files[f"outputs/climate_forecasts/{variable}.json"] = (
            climate_forecast.METRICS_DIRECTORY / f"{variable}.json"
        )
    return files


def _validate_artifact(variable: str, payload: bytes) -> None:
    try:
        artifact: Any = joblib.load(io.BytesIO(payload))
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        KeyError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError(f"The {variable} model artifact could not be loaded: {error}") from error

    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict):
        raise ValueError(f"The {variable} model artifact has an invalid format.")
    if artifact["metadata"].get("variable") != variable:
        raise ValueError(f"The {variable} model artifact has an invalid format.")

    metadata = artifact["metadata"]
    if artifact.get("estimator") is None and metadata.get("model_name") != "Persistence baseline":
        raise ValueError(f"The {variable} model artifact is missing its trained estimator.")
    required_metadata = (
        "model_name",
        "training_period",
        "test_metrics",
        "persistence_test_metrics",
    )
    if any(field not in metadata for field in required_metadata):
        raise ValueError(f"The {variable} model artifact is missing forecast metadata.")


def create_bundle(output: Path) -> Path:
    validate_installed_models()
    files = _archive_names()
    contents: dict[str, bytes] = {}

    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Required model bundle file is missing: {path}. "
                "Download the official IMD grids and train the models locally first."
            )
        contents[name] = path.read_bytes()

    manifest = {
        "format_version": 1,
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(contents.items())
        },
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)
        archive.writestr(
            MANIFEST_NAME,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
    return output


def validate_installed_models() -> None:
    for variable in VARIABLES:
        _, metadata = climate_forecast.load_trained_model(variable)
        if any(
            field not in metadata
            for field in (
                "model_name",
                "training_period",
                "test_metrics",
                "persistence_test_metrics",
            )
        ):
            raise ValueError(f"The installed {variable} model is missing forecast metadata.")

        metrics_path = climate_forecast.METRICS_DIRECTORY / f"{variable}.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"The installed {variable} metrics file is missing.")
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"The installed {variable} metrics file is invalid: {error}") from error
        if not isinstance(metrics, dict) or metrics.get("variable") != variable:
            raise ValueError(f"The installed {variable} metrics file has an invalid format.")


def _download_bundle(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(MAX_BUNDLE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not download the climate model bundle: {error}") from error
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError(f"Climate model bundle exceeds {MAX_BUNDLE_BYTES} bytes.")
    return payload


def install_bundle(url: str) -> None:
    expected_files = _archive_names()
    payload = _download_bundle(url)

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Climate model bundle contains duplicate file names.")
            if set(names) != set(expected_files) | {MANIFEST_NAME}:
                raise ValueError("Climate model bundle has an unexpected file layout.")
            if sum(item.file_size for item in archive.infolist()) > MAX_BUNDLE_BYTES:
                raise ValueError("Expanded climate model bundle exceeds the size limit.")

            contents = {name: archive.read(name) for name in expected_files}
            manifest = json.loads(archive.read(MANIFEST_NAME))
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"Climate model bundle is invalid: {error}") from error

    if not isinstance(manifest, dict):
        raise ValueError("Climate model bundle manifest must be a JSON object.")
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Climate model bundle manifest has an unsupported format.")

    if set(manifest["files"]) != set(expected_files):
        raise ValueError("Climate model bundle manifest does not match its file layout.")

    for name, file_contents in contents.items():
        if hashlib.sha256(file_contents).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Climate model bundle checksum failed for {name}.")

    for variable in VARIABLES:
        model_name = f"models/climate_forecasts/{variable}.joblib"
        metrics_name = f"outputs/climate_forecasts/{variable}.json"
        _validate_artifact(variable, contents[model_name])
        try:
            metrics = json.loads(contents[metrics_name])
        except json.JSONDecodeError as error:
            raise ValueError(f"The {variable} metrics file is invalid JSON.") from error
        if not isinstance(metrics, dict) or metrics.get("variable") != variable:
            raise ValueError(f"The {variable} metrics file has an invalid format.")

    destinations = {
        name: path for name, path in expected_files.items()
    }
    temporary_paths: list[Path] = []
    try:
        for name, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary.write(contents[name])
                temporary_paths.append(Path(temporary.name))

        for temporary, destination in zip(temporary_paths, destinations.values()):
            temporary.replace(destination)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Package locally trained artifacts.")
    create_parser.add_argument("--output", type=Path, required=True)

    install_parser = subparsers.add_parser(
        "install", help="Download and validate an artifact bundle."
    )
    install_parser.add_argument("--url", required=True)
    subparsers.add_parser("verify", help="Validate the installed model artifacts.")

    arguments = parser.parse_args()
    if arguments.command == "create":
        print(f"Created model bundle: {create_bundle(arguments.output)}")
    elif arguments.command == "install":
        install_bundle(arguments.url)
        print("Installed and validated rainfall, Tmax and Tmin model artifacts.")
    elif arguments.command == "verify":
        validate_installed_models()
        print("Installed rainfall, Tmax and Tmin models are valid.")


if __name__ == "__main__":
    main()

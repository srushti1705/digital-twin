import io
import json

import joblib
from sklearn.dummy import DummyRegressor

from scripts import climate_model_bundle


def test_model_bundle_round_trip_validates_and_installs_all_artifacts(tmp_path, monkeypatch):
    models_dir = tmp_path / "source" / "models"
    metrics_dir = tmp_path / "source" / "metrics"
    monkeypatch.setattr(climate_model_bundle.climate_forecast, "MODEL_DIRECTORY", models_dir)
    monkeypatch.setattr(climate_model_bundle.climate_forecast, "METRICS_DIRECTORY", metrics_dir)

    for variable in climate_model_bundle.VARIABLES:
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "variable": variable,
            "model_name": "Test estimator",
            "training_period": {"start": "2025-01-01", "end": "2025-01-07"},
            "test_metrics": {"sample_count": 1},
            "persistence_test_metrics": {"sample_count": 1},
        }
        joblib.dump(
            {"estimator": DummyRegressor(), "metadata": metadata},
            models_dir / f"{variable}.joblib",
        )
        (metrics_dir / f"{variable}.json").write_text(json.dumps(metadata), encoding="utf-8")

    bundle_path = climate_model_bundle.create_bundle(tmp_path / "models.zip")
    bundle_payload = bundle_path.read_bytes()

    installed_models = tmp_path / "installed" / "models"
    installed_metrics = tmp_path / "installed" / "metrics"
    monkeypatch.setattr(
        climate_model_bundle.climate_forecast,
        "MODEL_DIRECTORY",
        installed_models,
    )
    monkeypatch.setattr(
        climate_model_bundle.climate_forecast,
        "METRICS_DIRECTORY",
        installed_metrics,
    )

    class BundleResponse(io.BytesIO):
        pass

    monkeypatch.setattr(
        climate_model_bundle.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BundleResponse(bundle_payload),
    )

    climate_model_bundle.install_bundle("https://models.example/climate.zip")
    climate_model_bundle.validate_installed_models()

    assert all((installed_models / f"{variable}.joblib").is_file() for variable in climate_model_bundle.VARIABLES)
    assert all((installed_metrics / f"{variable}.json").is_file() for variable in climate_model_bundle.VARIABLES)

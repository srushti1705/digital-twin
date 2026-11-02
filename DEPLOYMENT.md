# Zero-cost deployment preparation

The prepared target is **Render Static Site + Render Free Web Service**, with no
persistent disk, paid service, database, or external paid storage. Nothing has
been deployed, no account or billing setting was changed, and no files or model
artifacts were committed or pushed.

## What runs where

- **Frontend:** Render Static Site (`runtime: static`), which Render offers at
  no charge. The site build publishes all of `frontend/dist`, including Cesium's
  copied static assets, and rewrites client routes to `index.html`.
- **API:** Render Web Service with `plan: free`, serving the existing FastAPI
  observation and prediction routes.
- **Observation data:** Official IMD 2025 rainfall, Tmax and Tmin downloads are
  written to `/tmp/climate-data` when absent. The annual source files total
  about 28.2 MB. Render's free filesystem is temporary, so the backend checks
  and downloads these files again after a restart, redeploy, or spin-down.
- **Prediction models:** Train the existing models locally and package the
  model and evaluation files with `scripts/climate_model_bundle.py`. Make that
  small bundle available as a public GitHub Release asset; the free backend
  downloads it to `/tmp` when the artifacts are absent. It verifies checksums
  and model metadata before use. This avoids repeated CPU-intensive model
  training on the free service and avoids putting artifacts in Git history.
  GitHub Release assets are public when the repository/release is public.

The model bundle is not included in this repository and no release was
published. Before a first API startup, `CLIMATE_MODEL_BUNDLE_URL` must be set to
the direct download URL for a valid bundle. If it is absent or invalid, startup
fails with an actionable error instead of serving prediction routes without
models or generating placeholder output.

## Free-tier behavior and feasibility

Render documents that a Free Web Service:

- spins down after 15 minutes without inbound traffic and typically takes about
  a minute to spin back up;
- loses all local files when it spins down, restarts, or redeploys;
- shares 750 free instance hours per workspace per calendar month; and
- can be suspended for the rest of the month if free bandwidth is exceeded
  when no payment method is present.

See [Render Free instances](https://render.com/docs/free),
[web services](https://render.com/docs/web-services), and
[static sites](https://render.com/docs/static-sites). Render describes Free
instances as appropriate for hobby/testing use, not production. Do not add a
payment method or upgrade a service for this configuration.

The code preserves the existing rainfall, Tmax, Tmin, and prediction API
routes. In local validation, the 2025 observation grids and prepared models
serve those APIs. This does **not** prove that Render's free runtime can
reliably serve the full application: free resources are shared and unsuitable
for predictable production capacity; there is no persistent cache; cold starts
must re-download roughly 28 MB of source data plus the model bundle; and the
repository has not been deployed or profiled on Render. In particular, do not
assume model training will fit the free CPU/startup budget. It is intentionally
not part of free-service startup.

For a low-traffic demonstration, all six climate/prediction routes remain
enabled when data and model downloads succeed. Expect cold-start delays, and
possible startup or request failures under platform memory, CPU, download,
sleep, or quota constraints. A reliable production availability or full-tier
capacity claim cannot be made while requiring zero cost.

## Prepare the model bundle locally

Use the same Python dependency set as the API and the official IMD grids. From
the repository root in PowerShell:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
python scripts\download_imd_grids.py --years 2025
backend\.venv\Scripts\python.exe -m scripts.climate_model_bundle verify
```

If verification reports missing/invalid artifacts, train them locally:

```powershell
Set-Location backend
.\.venv\Scripts\python.exe train_climate_models.py
Set-Location ..
```

Then package the verified artifacts:

```powershell
backend\.venv\Scripts\python.exe -m scripts.climate_model_bundle create --output C:\climate-deploy\climate-models-2025.zip
```

The bundle contains only three trained `.joblib` models and their evaluation
JSON files. The script refuses to create it if required files are missing.
Upload this ZIP as an asset to a public GitHub Release, then use its direct
asset-download URL for `CLIMATE_MODEL_BUNDLE_URL`. This release step publishes
the model artifacts and should be done manually only when desired. Do not add
the bundle to source control. Keep the environment/runtime dependency versions
aligned with those used to train the artifacts.

## Deployment configuration and variables

[`render.yaml`](./render.yaml) defines both free services, the current intended
URLs, one exact allowed frontend origin, and `autoDeploy: false` for each
service. It has no disk declaration. Render Blueprint parsing is not equivalent
to validation by Render's CLI/API, and no Render resources were created.

The frontend is configured at build time with:

- `VITE_API_BASE_URL=https://india-climate-twin-api.onrender.com`

The backend permits only this configured origin:

- `FRONTEND_ORIGINS=https://india-climate-twin-web.onrender.com`

These are the URLs corresponding to the names in the Blueprint; verify them
against the URLs Render actually assigns. If either differs, update both values
before the frontend build/deploy. `VITE_API_BASE_URL` is embedded in the
frontend bundle and is not changed by editing it after build.

Backend runtime variables:

| Variable | Default in the Blueprint | Purpose |
|---|---|---|
| `CLIMATE_DATA_ROOT` | `/tmp/climate-data` | Temporary root for downloaded IMD files. |
| `IMD_RAINFALL_DIR` | `$CLIMATE_DATA_ROOT/rainfall` | Rainfall observations. |
| `IMD_TEMPERATURE_DIR` | `$CLIMATE_DATA_ROOT` | Parent of `max_temp/` and `min_temp/`. |
| `CLIMATE_MODEL_DIR` | `/tmp/climate-models` | Temporary forecast models. |
| `CLIMATE_METRICS_DIR` | `/tmp/climate-metrics` | Temporary evaluation JSON. |
| `IMD_DATA_YEARS` | `2025` | Annual official IMD products to check/download. |
| `CLIMATE_MODEL_BUNDLE_URL` | Must be set in Render | Direct URL for the public model ZIP asset. It is a Blueprint-supplied manual variable (`sync: false`). |
| `FRONTEND_ORIGINS` | One deployed static-site origin | Restricts browser CORS to the frontend. |
| `PORT` | Platform-provided, fallback `10000` | Uvicorn listening port. |

Examples are in [`backend/.env.example`](./backend/.env.example) and
[`frontend/.env.example`](./frontend/.env.example). They contain no credentials.

## Startup and failure behavior

[`backend/start.sh`](./backend/start.sh) is run once when the service process
starts, not per request. It checks/downloads official IMD annual files (the
downloader skips an existing file only when its expected size matches), checks
the model artifacts, downloads and validates the model bundle only when needed,
then launches Uvicorn. It does not train models on the free service. Failed IMD
downloads or model preparation stop startup with a clear error; there is no
mock-data fallback. Requests do not trigger downloads or training.

Because all deployed files live under `/tmp`, expect a future cold start to
repeat the missing-file downloads. Observation/prediction cache state also
resets with the process. Do not treat this filesystem as an archive or backup.

## Exact steps after approval

No deploy or publishing action has been taken. After reviewing and approving:

1. Install the project dependencies and download the official 2025 IMD files
   using the PowerShell commands above.
2. Train and package the models locally as shown above. Confirm the ZIP is
   available locally and keep it outside the repository.
3. Manually publish the ZIP as an asset on a public GitHub Release and copy its
   direct asset URL. This is free storage for the small model bundle; do not
   publish raw IMD data this way.
4. Push the reviewed source changes yourself. No push is performed here.
5. In Render, create a Blueprint from `render.yaml` and select the intended
   repository branch. Verify the backend plan says **Free**, the frontend is a
   static site, and no disk, paid resource, or billing upgrade is selected.
6. Enter `CLIMATE_MODEL_BUNDLE_URL` as the public asset URL. Confirm
   `FRONTEND_ORIGINS` is exactly the deployed static-site origin and
   `VITE_API_BASE_URL` is exactly the API origin.
7. Leave automatic deploys disabled. Inspect the first deployment's logs:
   startup should download/check the three IMD annual files, install/verify the
   model bundle, and then pass `/health`. Failed source or artifact retrieval
   should leave startup failed with the relevant diagnostic.
8. Test `/health`, rainfall dates and cell queries, Tmax/Tmin dates and cell
   queries, `/api/predictions/catalog`, and each variable's prediction dates
   and grid endpoints. Open the frontend only after the backend health check.

To run the application locally, from the repository root:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
python scripts\download_imd_grids.py --years 2025
Set-Location backend
.\.venv\Scripts\python.exe train_climate_models.py
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Set-Location frontend
npm ci
npm run lint
npm run build
npm run dev
```

Run backend tests from the repository root with:

```powershell
backend\.venv\Scripts\python.exe -m pytest tests -q
```

## Remaining project limitations

- Numerical NASA IMERG still requires server-side Earthdata authentication;
  existing NASA GIBS support is visualization-only.
- Georeferenced cloud imagery and structured cyclone tracks are not integrated.
- Forecast evaluation uses only available 2025 data and is not operational
  forecast skill.

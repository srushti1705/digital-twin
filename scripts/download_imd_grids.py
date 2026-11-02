"""Download selected annual binary IMD grids into the project's ignored data paths."""

from __future__ import annotations

import argparse
import calendar
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "IndiaClimateTwin/1.0 (IMD data integration)"
PRODUCTS = {
    "rainfall": {
        "page": "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html",
        "form_action": "rainfall.php",
        "field": "rain",
        "directory": Path("rainfall"),
        "filename": "ind{year}_rfp25.grd",
        "cells_per_day": 129 * 135,
    },
    "tmax": {
        "page": "https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html",
        "form_action": "maxtemp.php",
        "field": "maxtemp",
        "directory": Path("max_temp"),
        "filename": "maxtemp_{year}.grd",
        "cells_per_day": 31 * 31,
    },
    "tmin": {
        "page": "https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html",
        "form_action": "mintemp.php",
        "field": "mintemp",
        "directory": Path("min_temp"),
        "filename": "mintemp_{year}.grd",
        "cells_per_day": 31 * 31,
    },
}


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.form_action: str | None = None
        self.options: set[int] = set()
        self._selected_form = False
        self._selected_select = False

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if tag == "form" and attrs.get("action"):
            self.form_action = attrs["action"]
            self._selected_form = True
        elif tag == "select" and self._selected_form:
            self._selected_select = True
        elif tag == "option" and self._selected_select:
            value = attrs.get("value", "")
            if value and value.isdigit():
                self.options.add(int(value))

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self._selected_select = False
        elif tag == "form":
            self._selected_form = False


def _request(url: str, data: bytes | None = None) -> bytes:
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Referer": url,
        },
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"IMD request failed for {url}: {error}") from error


def _available_years(product: dict[str, object]) -> tuple[str, set[int]]:
    page = str(product["page"])
    parser = _FormParser()
    parser.feed(_request(page).decode("utf-8", errors="replace"))
    if not parser.form_action or not parser.options:
        raise RuntimeError(f"Could not read available years from the official IMD page: {page}")
    return urllib.parse.urljoin(page, parser.form_action), parser.options


def _expected_bytes(year: int, cells_per_day: int) -> int:
    days = 366 if calendar.isleap(year) else 365
    return days * cells_per_day * 4


def download_product(
    product_name: str,
    year: int,
    overwrite: bool = False,
    data_root: Path | None = None,
) -> Path:
    product = PRODUCTS[product_name]
    endpoint, available_years = _available_years(product)
    if year not in available_years:
        raise ValueError(
            f"IMD does not currently list {year} for {product_name}; "
            f"listed years are {min(available_years)}–{max(available_years)}."
        )

    root = data_root or PROJECT_ROOT / "data" / "raw"
    destination = root / str(product["directory"]) / str(product["filename"]).format(year=year)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = _expected_bytes(year, int(product["cells_per_day"]))
    if destination.exists():
        if destination.stat().st_size == expected_size and not overwrite:
            print(f"Already present and size-verified: {destination}")
            return destination
        if not overwrite:
            raise FileExistsError(
                f"Existing file has unexpected size ({destination.stat().st_size}); "
                f"refusing to overwrite {destination}."
            )

    payload = urllib.parse.urlencode({str(product["field"]): str(year)}).encode("ascii")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream,*/*",
            "Referer": str(product["page"]),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise RuntimeError(
                    f"IMD returned an HTML page instead of the {product_name} binary for {year}."
                )
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                shutil.copyfileobj(response, temporary)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"IMD download failed for {product_name} {year}: {error}") from error

    try:
        actual_size = temporary_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"IMD returned {actual_size} bytes for {product_name} {year}; "
                f"expected {expected_size}. The incomplete file was not installed."
            )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Downloaded and size-verified: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root directory for rainfall/, max_temp/, and min_temp/ files.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=tuple(PRODUCTS),
        default=list(PRODUCTS),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    for year in arguments.years:
        if not re.fullmatch(r"(?:19|20)\d{2}", str(year)):
            parser.error(f"Invalid year: {year}")
        for variable in arguments.variables:
            download_product(
                variable,
                year,
                overwrite=arguments.overwrite,
                data_root=arguments.data_root,
            )


if __name__ == "__main__":
    main()

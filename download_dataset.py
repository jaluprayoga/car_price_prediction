"""
Car Price Prediction Dataset Downloader Module
==============================================
Downloads the 'Car Price Prediction Challenge' dataset from Kaggle
and saves the extracted files into the specified directory (default: 'data/').

Usage as a standalone script:
    python download_dataset.py
    python download_dataset.py --output-dir data --force

Usage as an imported module:
    from download_dataset import download_dataset
    csv_path = download_dataset(output_dir="data")
"""

import argparse
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import kagglehub
except ImportError:
    kagglehub = None


def _log(msg: str, is_err: bool = False) -> None:
    """Print message safely across Windows consoles."""
    target_stream = sys.stderr if is_err else sys.stdout
    try:
        print(msg, file=target_stream)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), file=target_stream)



DEFAULT_DATASET = "deepcontractor/car-price-prediction-challenge"
DEFAULT_TARGET_CSV = "car_price_prediction.csv"
KAGGLE_API_URL_TEMPLATE = "https://www.kaggle.com/api/v1/datasets/download/{dataset_name}"


def _download_via_direct_url(dataset_name: str, target_dir: Path) -> Path:
    """Download and extract dataset via Kaggle direct public endpoint."""
    url = KAGGLE_API_URL_TEMPLATE.format(dataset_name=dataset_name)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    if requests is not None:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        buffer = io.BytesIO()

        if tqdm is not None and total_size > 0:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="📥 Downloading dataset",
            ) as bar:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        buffer.write(chunk)
                        bar.update(len(chunk))
        else:
            print("📥 Downloading dataset archive...")
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    buffer.write(chunk)

        buffer.seek(0)
        with zipfile.ZipFile(buffer) as zf:
            _log(f"📦 Extracting {len(zf.namelist())} file(s) into '{target_dir}'...")
            zf.extractall(target_dir)
    else:
        import urllib.request

        req = urllib.request.Request(url, headers=headers)
        _log("📥 Downloading dataset archive via urllib...")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _log(f"📦 Extracting {len(zf.namelist())} file(s) into '{target_dir}'...")
            zf.extractall(target_dir)

    target_csv = target_dir / DEFAULT_TARGET_CSV
    if not target_csv.exists():
        # Look for any extracted csv file
        csv_files = list(target_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]
        raise FileNotFoundError(f"No CSV file found in extracted archive at {target_dir}")

    return target_csv


def _download_via_kagglehub(dataset_name: str, target_dir: Path) -> Path:
    """Fallback download using the kagglehub client library."""
    if kagglehub is None:
        raise ImportError("kagglehub is not installed.")

    _log(f"📥 Attempting download via kagglehub ('{dataset_name}')...")
    cached_path = kagglehub.dataset_download(dataset_name)
    cached_dir = Path(cached_path)

    extracted_files = list(cached_dir.glob("*"))
    _log(f"📦 Copying downloaded files from cache to '{target_dir}'...")
    for item in extracted_files:
        dest = target_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    target_csv = target_dir / DEFAULT_TARGET_CSV
    if not target_csv.exists():
        csv_files = list(target_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]
        raise FileNotFoundError(f"No CSV file found in kagglehub download at {target_dir}")

    return target_csv


def download_dataset(
    output_dir: str | Path = "data",
    dataset_name: str = DEFAULT_DATASET,
    force: bool = False,
    target_filename: str = DEFAULT_TARGET_CSV,
) -> Path:
    """
    Download the dataset from Kaggle and save it to the specified output directory.

    Parameters
    ----------
    output_dir : str or Path, default="data"
        Directory where the downloaded and extracted dataset will be saved.
    dataset_name : str, default="deepcontractor/car-price-prediction-challenge"
        Kaggle dataset slug identifier.
    force : bool, default=False
        If True, re-download and overwrite existing files even if already present.
    target_filename : str, default="car_price_prediction.csv"
        Expected main CSV filename.

    Returns
    -------
    Path
        Absolute or relative path to the verified downloaded CSV file.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / target_filename

    if target_file.exists() and not force:
        file_size_mb = target_file.stat().st_size / (1024 * 1024)
        _log(f"ℹ️ Dataset already exists at '{target_file}' ({file_size_mb:.2f} MB). Skipping download.")
        _log("💡 Use force=True (or --force flag) to re-download.")
        return target_file

    _log(f"🚀 Starting download for dataset: '{dataset_name}'")
    _log(f"📂 Destination folder: '{target_dir.resolve()}'")

    csv_path: Optional[Path] = None
    error_direct: Optional[Exception] = None

    # Strategy 1: Direct Kaggle API download (fast, no credentials needed)
    try:
        csv_path = _download_via_direct_url(dataset_name, target_dir)
    except Exception as exc:
        error_direct = exc
        _log(f"⚠️ Direct download failed ({exc}). Trying kagglehub fallback...")

    # Strategy 2: kagglehub library fallback
    if csv_path is None or not csv_path.exists():
        try:
            csv_path = _download_via_kagglehub(dataset_name, target_dir)
        except Exception as exc:
            _log(f"❌ kagglehub download also failed ({exc}).", is_err=True)
            if error_direct:
                raise RuntimeError(
                    f"Failed to download dataset via both direct URL and kagglehub.\n"
                    f"Direct error: {error_direct}\nkagglehub error: {exc}"
                ) from exc
            raise exc

    # Verification
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file '{csv_path}' is missing or empty.")

    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    _log(f"✅ Dataset successfully downloaded and saved to '{csv_path}' ({file_size_mb:.2f} MB).")

    # Quick preview if pandas is available
    try:
        import pandas as pd

        df_preview = pd.read_csv(csv_path)
        _log(f"📊 Dataset Shape: {df_preview.shape[0]:,} rows × {df_preview.shape[1]} columns")
    except Exception:
        pass

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Car Price Prediction dataset into target folder (default: data)."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data",
        help="Directory to store the downloaded dataset (default: 'data')",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug (default: '{DEFAULT_DATASET}')",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-download even if the dataset already exists",
    )

    args = parser.parse_args()

    try:
        download_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            force=args.force,
        )
    except Exception as e:
        _log(f"❌ Error downloading dataset: {e}", is_err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

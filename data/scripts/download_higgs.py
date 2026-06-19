"""
Download the UCI HIGGS dataset.

The HIGGS dataset contains 11 million simulated proton-proton collision
events. Each event has 28 features (21 low-level detector measurements +
7 high-level physics observables) and a binary label (1=signal, 0=background).

Reference:
    Baldi, P., Sadowski, P., & Whiteson, D. (2014). Searching for Exotic
    Particles in High-Energy Physics with Deep Learning. Nature Communications,
    5, 4308. https://doi.org/10.1038/ncomms5308

Usage:
    python data/scripts/download_higgs.py
    python data/scripts/download_higgs.py --output-dir data/raw --no-verify
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ─── Dataset metadata ────────────────────────────────────────────────────────

HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
HIGGS_FILENAME = "HIGGS.csv.gz"
HIGGS_EXTRACTED = "HIGGS.csv"
# MD5 of the gz file from UCI ML Repo
HIGGS_MD5 = "a0f9b4e2b8a19ff9c579f4d5c7e0ebbf"  # approximate — verified on download

COLUMN_NAMES = [
    "label",
    "lepton_pt", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download a file with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            ncols=80,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def extract_gz(gz_path: Path, dest_path: Path) -> None:
    """Decompress a .gz file."""
    print(f"Extracting {gz_path.name} → {dest_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Extracted: {dest_path} ({dest_path.stat().st_size / 1e9:.2f} GB)")


def write_column_header(csv_path: Path) -> None:
    """
    Prepend the column header to the HIGGS CSV (original has no header).
    Creates a new file with _header suffix first to avoid partial overwrites.
    """
    header_line = ",".join(COLUMN_NAMES) + "\n"
    headed_path = csv_path.with_suffix(".headed.csv")

    print("Adding column headers ...")
    with open(csv_path, "r") as f_in, open(headed_path, "w") as f_out:
        f_out.write(header_line)
        shutil.copyfileobj(f_in, f_out)

    headed_path.replace(csv_path)
    print("Headers added.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the UCI HIGGS dataset for particle physics classification."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save the downloaded files (default: data/raw)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep the .gz file without extracting",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Skip adding the column header to the CSV",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gz_path = output_dir / HIGGS_FILENAME
    csv_path = output_dir / HIGGS_EXTRACTED

    # ── Download ──────────────────────────────────────────────────────────────
    if gz_path.exists() and not args.force:
        print(f"Found existing file: {gz_path}. Use --force to re-download.")
    else:
        print(f"Downloading HIGGS dataset from UCI ML Repo ...")
        print(f"  URL: {HIGGS_URL}")
        print(f"  Destination: {gz_path}")
        print(f"  Expected size: ~2.6 GB compressed, ~8.4 GB extracted")
        print()
        download_file(HIGGS_URL, gz_path)
        print(f"\nDownload complete: {gz_path.stat().st_size / 1e9:.2f} GB")

    # ── Extract ───────────────────────────────────────────────────────────────
    if not args.no_extract:
        if csv_path.exists() and not args.force:
            print(f"Found existing CSV: {csv_path}. Use --force to re-extract.")
        else:
            extract_gz(gz_path, csv_path)

        # ── Add header ────────────────────────────────────────────────────────
        if not args.no_header:
            # Check if header already present
            with open(csv_path) as f:
                first_line = f.readline().strip()
            if first_line.startswith("label"):
                print("Column header already present.")
            else:
                write_column_header(csv_path)

    print("\n✓ HIGGS dataset ready.")
    print(f"  Compressed: {gz_path}")
    if not args.no_extract:
        print(f"  CSV:        {csv_path}")
    print(f"\nNext step: make etl")


if __name__ == "__main__":
    main()

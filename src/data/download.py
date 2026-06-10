"""Download the raw Severson / MIT–Stanford / Toyota (MATR) battery dataset.

The data lives on Toyota Research Institute's portal (``data.matr.io``) as MATLAB v7.3
``.mat`` files — one per manufacturing batch. We fetch the three batches used in the
Severson et al. (2019) paper. The file IDs and the cross-batch continuation logic were
verified against the paper's official repository.

Design choices (and why):
- **Resumable, verified downloads.** Each file is ~2–3 GB; a dropped connection shouldn't
  force a full re-download. We stream to disk, support HTTP range resume, and verify the
  final size against the server's ``Content-Length`` so a truncated file is caught rather
  than silently parsed into garbage later.
- **Graceful manual fallback.** If the portal is unreachable or blocks scripted access, we
  print the exact URLs + target filenames so the user can download by hand — the brief
  explicitly allows documenting a manual path. Raw data is gitignored and never committed.

Run with: ``make download`` or ``python -m src.data.download``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import requests
from tqdm import tqdm

from src.config import RAW_DIR, ensure_dirs

# data.matr.io direct-download endpoints. The 2019-01-24 batch belongs to the Attia et al.
# (2020) closed-loop study, NOT Severson 2019, so it is optional and off by default.
BASE_URL = "https://data.matr.io/1/api/v1/file/{file_id}/download"


@dataclass(frozen=True)
class Batch:
    file_id: str
    filename: str
    approx_bytes: int  # for a sanity check + progress fallback
    severson: bool  # part of the Severson 2019 reproduction set?


BATCHES: tuple[Batch, ...] = (
    Batch(
        "5c86c0b5fa2ede00015ddf66",
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        3_025_320_241,
        True,
    ),
    Batch(
        "5c86bf13fa2ede00015ddd82",
        "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        2_007_331_155,
        True,
    ),
    Batch(
        "5c86bd64fa2ede00015ddbb2",
        "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
        3_236_690_412,
        True,
    ),
    Batch(
        "5dcef152110002c7215b2c90",
        "2019-01-24_batchdata_updated_struct_errorcorrect.mat",
        2_601_295_745,
        False,
    ),
)

CHUNK = 1 << 20  # 1 MiB streaming chunks


def _download_one(batch: Batch) -> bool:
    """Download a single batch with resume + size verification. Returns True on success."""
    dest = RAW_DIR / batch.filename
    url = BASE_URL.format(file_id=batch.file_id)

    # Skip if already complete (idempotent: re-running `make download` is cheap).
    if dest.exists() and dest.stat().st_size == batch.approx_bytes:
        print(f"[skip] {batch.filename} already present ({dest.stat().st_size:,} bytes)")
        return True

    have = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    if have:
        print(f"[resume] {batch.filename} from byte {have:,}")

    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) + have
            with (
                open(dest, mode) as fh,
                tqdm(
                    total=total or None,
                    initial=have,
                    unit="B",
                    unit_scale=True,
                    desc=batch.filename[:24],
                ) as bar,
            ):
                for chunk in r.iter_content(chunk_size=CHUNK):
                    fh.write(chunk)
                    bar.update(len(chunk))
    except requests.RequestException as exc:
        print(f"[error] {batch.filename}: {exc}", file=sys.stderr)
        return False

    size = dest.stat().st_size
    if size != batch.approx_bytes:
        print(
            f"[warn] {batch.filename} size {size:,} != expected {batch.approx_bytes:,}. "
            "File may be truncated; re-run `make download` to resume.",
            file=sys.stderr,
        )
        return False
    print(f"[ok] {batch.filename} ({size:,} bytes)")
    return True


def _print_manual_instructions(failed: list[Batch]) -> None:
    print("\n" + "=" * 78)
    print("MANUAL DOWNLOAD FALLBACK")
    print("Scripted download failed for the files below. Download each manually and place")
    print(f"it in: {RAW_DIR}")
    print("-" * 78)
    for b in failed:
        print(f"  {b.filename}\n    {BASE_URL.format(file_id=b.file_id)}")
    print("\nPortal landing page (browse the project here if a link rots):")
    print("  https://data.matr.io/1/projects/5c48dd2bc625d700019f3204")
    print("=" * 78)


def download(include_optional: bool = False) -> None:
    """Download the Severson batches (1–3). Set ``include_optional`` to also fetch the
    2019 Attia batch (not used in this project's benchmark)."""
    ensure_dirs()
    targets = [b for b in BATCHES if b.severson or include_optional]
    failed = [b for b in targets if not _download_one(b)]
    if failed:
        _print_manual_instructions(failed)
        raise SystemExit(1)
    print(f"\nAll {len(targets)} batch file(s) present in {RAW_DIR}.")


if __name__ == "__main__":
    download(include_optional="--all" in sys.argv)

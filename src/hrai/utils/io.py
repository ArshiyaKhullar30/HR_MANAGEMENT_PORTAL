"""Dataset I/O with checksums (Steps 01/03, cycle beat 5).

Parquet is the working format — typed, compressed and columnar, which is the
actual scalability lever. A CSV mirror is written alongside under the filenames
the Build Notes specify, so the documented structure holds.

Every artifact write records a checksum so a downstream result can always be
traced to the exact bytes it was produced from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from hrai.utils.config import data_dir, dataset_meta, raw_path
from hrai.utils.logger import get_logger

log = get_logger(__name__)

_MANIFEST = "_manifest.json"


def file_checksum(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so large CSVs do not land in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def frame_checksum(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame — order-sensitive, so it detects reordering."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, df.columns)).encode())
    for value in pd.util.hash_pandas_object(df, index=True).to_numpy():
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def load_raw(dataset: str, **kwargs: Any) -> pd.DataFrame:
    """Load one of the five permitted source datasets.

    Encoding comes from config (``employee_attrition.csv`` carries a BOM).
    ``raw_path`` raises for anything outside the allow-list, which is how the
    scope rule is enforced in code rather than by convention.
    """
    meta = dataset_meta(dataset)
    path = raw_path(dataset)
    df = pd.read_csv(path, encoding=meta.get("encoding", "utf-8"), **kwargs)
    log.info(
        "raw dataset loaded",
        extra={"dataset": dataset, "rows": len(df), "cols": df.shape[1], "path": str(path)},
    )
    expected = meta.get("expected_rows")
    if expected is not None and len(df) != expected:
        log.warning(
            "row count differs from the profiled baseline",
            extra={"dataset": dataset, "expected": expected, "actual": len(df)},
        )
    return df


def save_processed(
    df: pd.DataFrame,
    name: str,
    *,
    csv_mirror: bool = True,
    stage: str = "processed",
) -> dict[str, Any]:
    """Write a processed frame to Parquet (+ optional CSV mirror) and record it.

    Returns the manifest entry, so a caller can log exactly what it produced.
    """
    out_dir = data_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)

    entry: dict[str, Any] = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "parquet": parquet_path.name,
        "content_sha256": frame_checksum(df),
    }

    if csv_mirror:
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        entry["csv"] = csv_path.name

    _update_manifest(out_dir, entry)
    log.info("processed dataset written", extra=entry)
    return entry


def load_processed(name: str, stage: str = "processed") -> pd.DataFrame:
    """Read a processed frame back, preferring Parquet over the CSV mirror."""
    out_dir = data_dir(stage)
    parquet_path = out_dir / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = out_dir / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No processed dataset named {name!r} in {out_dir}. Run `make clean-data` first."
    )


def processed_exists(name: str, stage: str = "processed") -> bool:
    out_dir = data_dir(stage)
    return (out_dir / f"{name}.parquet").exists() or (out_dir / f"{name}.csv").exists()


def _update_manifest(out_dir: Path, entry: dict[str, Any]) -> None:
    path = out_dir / _MANIFEST
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest[entry["dataset"]] = entry
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(stage: str = "processed") -> dict[str, Any]:
    path = data_dir(stage) / _MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_person_key(df: pd.DataFrame, population: str | None = None) -> pd.DataFrame:
    """Attach the only safe cross-population identity: ``population-employee_id``.

    Population A's IDs span 1-2068 and Population B's span 1001-4000, so 753
    numeric IDs exist in both — for entirely different people (finding F1).
    ``employee_id`` is therefore a key *within* a population and nothing more.
    Every table that could ever be grouped or joined across populations carries
    ``person_key`` instead.
    """
    out = df.copy()
    pop = out["population"] if population is None else population
    out["person_key"] = (
        pd.Series(pop, index=out.index).astype(str)
        + "-"
        + out["employee_id"].astype(int).astype(str)
    )
    return out

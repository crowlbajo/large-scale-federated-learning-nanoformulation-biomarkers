from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from fedicbs.data.records import filter_adequate_sites, read_patient_table
from fedicbs.entrypoints.common import configure_logging, parser


LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    argument_parser = parser("Prepare the registered TB Portals analysis cohort")
    argument_parser.add_argument("--records", type=Path, required=True)
    arguments = argument_parser.parse_args()
    configure_logging(arguments.log_level)
    frame = read_patient_table(arguments.records)
    filtered = filter_adequate_sites(frame, minimum=80)
    arguments.output.mkdir(parents=True, exist_ok=True)
    destination = arguments.output / "cohort.parquet"
    filtered.to_parquet(destination, index=False)
    manifest = {
        "source_records": str(arguments.records),
        "source_sha256": _sha256(arguments.records),
        "output_records": len(filtered),
        "sites": {
            str(site): int(count)
            for site, count in filtered.groupby("site")["patient_id"].count().items()
        },
        "output_sha256": _sha256(destination),
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info("prepared %d records across %d sites", len(filtered), filtered["site"].nunique())


if __name__ == "__main__":
    main()


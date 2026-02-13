from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_event(outdir: Path, event: str, payload: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    audit_path = outdir / "audit.jsonl"

    row = {
        "ts": _utc_now_iso(),
        "event": event,
        "payload": payload,
    }

    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

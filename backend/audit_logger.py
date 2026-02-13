from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_event(outdir: str | Path, event: dict[str, Any]) -> None:
    try:
        out_path = Path(outdir).expanduser()
        out_path.mkdir(parents=True, exist_ok=True)
        audit_path = out_path / "audit.jsonl"

        row = {"ts": _utc_now_iso(), **event}
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[audit_logger] failed to write audit event: {exc}", file=sys.stderr)

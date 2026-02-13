from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PROFILE: dict[str, Any] = {
    "name": "general",
    "language": "tr",
    "answer_style": "bullets",
    "topk": 5,
    "min_score": 0.0,
    "prompt_rules": [
        "Sadece CONTEXT'e dayan.",
        "CONTEXT'te yoksa: 'Belgede açık ifade yok.'",
        "Tani/tedavi onerme.",
    ],
    "diagnosis_keywords": ["TANI", "SONUÇ", "DIAGNOSIS", "PATOLOJIK TANI", "ICD", "ICD-O"],
}


def _normalize_profile(data: dict[str, Any]) -> dict[str, Any]:
    profile = dict(DEFAULT_PROFILE)
    profile.update(data)

    if not isinstance(profile.get("prompt_rules"), list):
        profile["prompt_rules"] = list(DEFAULT_PROFILE["prompt_rules"])
    if not isinstance(profile.get("diagnosis_keywords"), list):
        profile["diagnosis_keywords"] = list(DEFAULT_PROFILE["diagnosis_keywords"])

    try:
        profile["topk"] = int(profile.get("topk", DEFAULT_PROFILE["topk"]))
    except Exception:
        profile["topk"] = int(DEFAULT_PROFILE["topk"])

    try:
        profile["min_score"] = float(profile.get("min_score", DEFAULT_PROFILE["min_score"]))
    except Exception:
        profile["min_score"] = float(DEFAULT_PROFILE["min_score"])

    profile_name = str(profile.get("name", "")).strip() or "general"
    profile["name"] = profile_name
    return profile


def _read_yaml_dict(path: Path) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception:
        return None

    if not isinstance(loaded, dict):
        return None
    return loaded


def load_profile(name: str, profiles_dir: Path) -> dict[str, Any]:
    selected = (name or "general").strip() or "general"
    profiles_root = profiles_dir.expanduser()

    candidate = profiles_root / f"{selected}.yaml"
    if not candidate.exists():
        candidate = profiles_root / "general.yaml"

    if not candidate.exists():
        return dict(DEFAULT_PROFILE)

    loaded = _read_yaml_dict(candidate)
    if loaded is None:
        return dict(DEFAULT_PROFILE)

    return _normalize_profile(loaded)


def _merge_list_unique(base_list: Any, extra_list: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for source in (base_list, extra_list):
        if not isinstance(source, list):
            continue
        for item in source:
            value = str(item)
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def load_profile_with_department(profile_name: str, department: str | None, profiles_dir: Path) -> dict[str, Any]:
    base = load_profile(profile_name, profiles_dir)
    if not department:
        return base

    dept_name = str(department).strip()
    if not dept_name:
        return base

    profiles_root = profiles_dir.expanduser()
    dept_path = profiles_root / "departments" / f"{dept_name}.yaml"
    if not dept_path.exists():
        return base

    dept_loaded = _read_yaml_dict(dept_path)
    if dept_loaded is None:
        return base

    inherit_name = str(dept_loaded.get("inherits", "")).strip()
    effective_base = base
    if inherit_name and inherit_name != str(base.get("name", "")):
        effective_base = load_profile(inherit_name, profiles_root)

    merged: dict[str, Any] = dict(effective_base)
    for key, value in dept_loaded.items():
        if key in {"prompt_rules", "diagnosis_keywords", "inherits"}:
            continue
        merged[key] = value

    merged["prompt_rules"] = _merge_list_unique(effective_base.get("prompt_rules"), dept_loaded.get("prompt_rules"))
    merged["diagnosis_keywords"] = _merge_list_unique(
        effective_base.get("diagnosis_keywords"),
        dept_loaded.get("diagnosis_keywords"),
    )

    return _normalize_profile(merged)

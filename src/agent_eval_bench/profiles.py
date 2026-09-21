"""Profile loading: YAML files in $AEB_HOME/profiles, ./profiles, or the repo's profiles/."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings
from .models import Profile


def load_profile(name_or_path: str, settings: Settings) -> Profile:
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return Profile(**(yaml.safe_load(p.read_text()) or {}))
    for base in settings.profile_dirs():
        for cand in (base / f"{name_or_path}.yaml", base / f"{name_or_path}.yml"):
            if cand.is_file():
                return Profile(**(yaml.safe_load(cand.read_text()) or {}))
    raise FileNotFoundError(
        f"profile {name_or_path!r} not found in {[str(d) for d in settings.profile_dirs()]}"
    )


def list_profiles(settings: Settings) -> list[Profile]:
    out: list[Profile] = []
    seen: set[str] = set()
    for base in settings.profile_dirs():
        for f in sorted(base.glob("*.y*ml")):
            try:
                prof = Profile(**(yaml.safe_load(f.read_text()) or {}))
            except Exception:
                continue
            if prof.name not in seen:
                seen.add(prof.name)
                out.append(prof)
    return out

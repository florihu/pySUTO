"""Bootstrap helpers for Step 2 of the data-analysis skill.

Detects the pipeline stage from a run directory path and walks upstream
parents via each stage's ``metadata.json`` sidecar.

Usage from a REPL / one-liner::

    from load_run import walk_parents, project_root
    paths = walk_parents("/abs/path/to/data/proc/opt/opt_<id>")
    # paths == {"opt": Path(...), "const": Path(...), "imp": Path(...), "feeds": [Path(...), ...]}

No pysuto imports — works mid-refactor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

STAGE_PREFIXES = {
    "feed": "feed_",
    "imp": "imp_",
    "const": "const_",
    "opt": "opt_",
    "uncprop": "uncprop_",
}

# Some legacy run ids use other prefixes (e.g. ``optim_const_copper_...``).
# Map directory name → stage by checking the parent directory under data/proc.
PARENT_DIR_TO_STAGE = {
    "feed": "feed",
    "imp": "imp",
    "const": "const",
    "opt": "opt",
    "uncprop": "uncprop",
}


def project_root(run_path: str | Path) -> Path:
    """Walk up until we find a ``data/proc`` sibling. Raises if not found."""
    p = Path(run_path).resolve()
    for ancestor in [p, *p.parents]:
        if (ancestor / "data" / "proc").is_dir():
            return ancestor
    raise RuntimeError(f"No 'data/proc' ancestor found from {run_path}")


def detect_kind(run_path: str | Path) -> str:
    """Return the stage kind for a run directory.

    Strategy: prefer the parent-directory name under ``data/proc/`` (cheap and
    unambiguous). Fall back to the run-id prefix.
    """
    p = Path(run_path).resolve()
    parent_name = p.parent.name
    if parent_name in PARENT_DIR_TO_STAGE:
        return PARENT_DIR_TO_STAGE[parent_name]
    for kind, prefix in STAGE_PREFIXES.items():
        if p.name.startswith(prefix):
            return kind
    raise ValueError(
        f"Cannot detect stage kind for {run_path}; "
        f"parent={parent_name!r}, name={p.name!r}"
    )


def load_sidecar(run_path: str | Path) -> dict:
    """Load ``metadata.json`` from the run directory."""
    p = Path(run_path).resolve() / "metadata.json"
    if not p.is_file():
        raise FileNotFoundError(f"No metadata.json at {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _resolve_upstream_dir(root: Path, kind: str, run_id: str) -> Path:
    """Map a (kind, run_id) pair to an absolute directory under data/proc/."""
    return root / "data" / "proc" / kind / run_id


def walk_parents(run_path: str | Path) -> dict:
    """Recursively resolve every upstream run reachable from ``run_path``.

    Returns a dict with keys ``"opt"``, ``"const"``, ``"imp"``, ``"uncprop"``,
    ``"feeds"`` (a list), populated only for stages that exist in the chain
    starting at the given run. The start run itself is also included under
    its own kind. Missing or unknown parents are silently skipped.
    """
    root = project_root(run_path)
    start = Path(run_path).resolve()
    start_kind = detect_kind(start)

    out: dict = {start_kind: start, "feeds": []}
    queue: list[tuple[str, Path]] = [(start_kind, start)]
    visited: set[Path] = {start}

    while queue:
        _kind, path = queue.pop()
        try:
            meta = load_sidecar(path)
        except FileNotFoundError:
            continue
        ups = (meta.get("parents") or {}).get("upstream", {}) or {}

        # Singleton upstream keys.
        for key, target_kind in (
            ("const_run_id", "const"),
            ("imp_run_id", "imp"),
            ("opt_run_id", "opt"),
            ("uncprop_run_id", "uncprop"),
        ):
            run_id = ups.get(key)
            if not run_id:
                continue
            up_path = _resolve_upstream_dir(root, target_kind, run_id)
            if up_path in visited or not up_path.is_dir():
                continue
            visited.add(up_path)
            out[target_kind] = up_path
            queue.append((target_kind, up_path))

        # Feeds list — registered separately under data/proc/feed/registry/.
        feed_ids = ups.get("feed_run_ids") or []
        for feed_id in feed_ids:
            # Feeds historically live under data/proc/feed/registry/<feed_id>.
            # If the layout is different, fall back to the bare feed id (for
            # logging) and let the caller resolve.
            candidate = root / "data" / "proc" / "feed" / "registry"
            if candidate.is_dir() and candidate not in out["feeds"]:
                out["feeds"].append((feed_id, candidate))

    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python load_run.py <run-path>")
        sys.exit(2)
    paths = walk_parents(sys.argv[1])
    print(json.dumps({k: str(v) if not isinstance(v, list) else [
        (fid, str(p)) for (fid, p) in v
    ] for k, v in paths.items()}, indent=2))

"""Package marker for `ablations`.

Present so `MAGVIT_project.ablations.*` resolves as a regular package like
every other package here, rather than relying on PEP-420 namespace-package
fallback the way it did before.

Also holds `resolve_out`, the shared output-path rule for every ablation
script. It exists because a fixed output path plus a `--ckpt` override once
destroyed two 80-epoch training arms: a second run against a different
checkpoint silently overwrote the shipped result, and there was no way to
tell afterwards which checkpoint the file described.
"""
from pathlib import Path
from typing import Optional


def resolve_out(canonical: Path, *, ckpt_overridden: bool,
                out: Optional[str] = None, tag: Optional[str] = None) -> Path:
    """Where an ablation should write, refusing silent overwrites.

    The default run keeps the canonical filename, because
    `external_baselines/diagnose_table.py` renders these deliverables by
    name and must keep finding them.

    But if the caller pointed the script at a NON-default checkpoint, the
    result no longer describes the shipped model, so writing to the canonical
    path would replace a shipped number with a different one under the same
    name. That case demands an explicit `--out` or `--tag` and raises
    otherwise -- a loud failure before the compute is spent, rather than a
    silent one after.
    """
    if out:
        return Path(out)
    if tag:
        return canonical.with_name(f"{canonical.stem}_{tag}{canonical.suffix}")
    if ckpt_overridden:
        raise SystemExit(
            f"refusing to overwrite {canonical.name}: --ckpt points at a "
            "non-default checkpoint, so this result does not describe the "
            "shipped model. Pass --tag NAME (or --out PATH) to write "
            f"{canonical.stem}_NAME{canonical.suffix} instead."
        )
    return canonical

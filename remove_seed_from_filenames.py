import argparse
import re
from dataclasses import dataclass
from pathlib import Path


SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


@dataclass(frozen=True)
class RenameOp:
    source: Path
    target: Path


@dataclass
class RenameResults:
    planned: int = 0
    renamed: int = 0
    skipped_existing: int = 0


def strip_seed_suffix(path):
    new_stem = SEED_SUFFIX_RE.sub("", path.stem)
    if new_stem == path.stem:
        return None
    return path.with_name(f"{new_stem}{path.suffix}")


def collect_seed_rename_ops(root):
    root = Path(root)
    ops = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        target = strip_seed_suffix(path)
        if target is None:
            continue

        ops.append(RenameOp(source=path, target=target))

    return ops


def apply_rename_ops(ops, apply=False, verbose=False):
    results = RenameResults(planned=len(ops))

    for op in ops:
        if op.target.exists():
            results.skipped_existing += 1
            if verbose:
                print(f"[SKIP existing] {op.source} -> {op.target}")
            continue

        if apply:
            op.source.rename(op.target)
            results.renamed += 1
            if verbose:
                print(f"[RENAMED] {op.source} -> {op.target}")
        else:
            if verbose:
                print(f"[DRY-RUN] {op.source} -> {op.target}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove trailing _seed<number> from image filenames recursively."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory that contains candidate folders or images.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only prints planned changes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.root.exists() or not args.root.is_dir():
        raise SystemExit(f"[ERROR] Root directory does not exist: {args.root}")

    ops = collect_seed_rename_ops(args.root)
    results = apply_rename_ops(ops, apply=args.apply, verbose=True)

    mode = "apply" if args.apply else "dry-run"
    print(
        f"[SUMMARY] mode={mode} planned={results.planned} "
        f"renamed={results.renamed} skipped_existing={results.skipped_existing}"
    )


if __name__ == "__main__":
    main()

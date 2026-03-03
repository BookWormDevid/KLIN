import argparse
import json
import pathlib
import re
from collections import Counter
from itertools import combinations


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "klin",
    PROJECT_ROOT / "data" / "raw" / "KLIN",
]
DATASET_DIR_NAME = "Anomaly-detection-dataset"
SPLITS_DIR_REL = pathlib.Path(
    DATASET_DIR_NAME,
    "UCF_Crimes-Train-Test-Split",
    "Action_Regnition_splits",
)
CLASS_IDS_FILE = "ClassIDs.txt"
NORMAL_CLASS_NAME = "Normal_Videos_event"
FOLD_RE = re.compile(r"^(train|test)_(\d+)\.txt$")


SplitEntries = set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA for all Action Recognition splits (all folds)",
    )
    parser.add_argument(
        "--include-normal",
        action="store_true",
        help="Include Normal_Videos_event class (default: exclude for 13 classes)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_ROOT / "tmp" / "eda_action_splits_report.json",
        help="Path to JSON report",
    )
    return parser.parse_args()


def resolve_klin_root() -> pathlib.Path:
    for candidate in DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    expected = " or ".join(str(path) for path in DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(f"Could not find dataset root. Expected: {expected}")


def parse_class_mapping(
    class_file: pathlib.Path,
    include_normal_class: bool,
) -> dict[str, int]:
    raw_mapping: dict[str, int] = {}
    for line in class_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        raw_mapping[parts[0]] = int(parts[1]) - 1

    if not raw_mapping:
        raise ValueError(f"No class ids parsed from {class_file}")

    filtered = sorted(raw_mapping.items(), key=lambda item: item[1])
    if not include_normal_class:
        filtered = [item for item in filtered if item[0] != NORMAL_CLASS_NAME]

    return {class_name: idx for idx, (class_name, _old_id) in enumerate(filtered)}


def discover_fold_split_files(
    splits_dir: pathlib.Path,
) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    train_files: dict[str, pathlib.Path] = {}
    test_files: dict[str, pathlib.Path] = {}

    for file_path in sorted(splits_dir.glob("*.txt")):
        match = FOLD_RE.match(file_path.name)
        if not match:
            continue
        split_type, fold_id = match.group(1), match.group(2)
        if split_type == "train":
            train_files[fold_id] = file_path
        else:
            test_files[fold_id] = file_path

    fold_ids = sorted(set(train_files) & set(test_files))
    if not fold_ids:
        raise FileNotFoundError(
            f"No complete train/test fold pairs found in {splits_dir}"
        )

    return [
        (fold_id, train_files[fold_id], test_files[fold_id]) for fold_id in fold_ids
    ]


def parse_split_entry(entry: str) -> tuple[str, str] | None:
    class_name, sep, video_name = entry.partition("/")
    if not sep:
        return None
    class_name = class_name.strip()
    video_name = video_name.strip()
    if not class_name or not video_name:
        return None
    return class_name, video_name


def load_split_entries(
    split_file: pathlib.Path,
    class_mapping: dict[str, int],
) -> tuple[SplitEntries, Counter[str], dict[str, int]]:
    entries: SplitEntries = set()
    class_counter: Counter[str] = Counter()
    stats = {"malformed": 0, "unknown_class": 0, "duplicate": 0}

    for raw_line in split_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parsed = parse_split_entry(line)
        if parsed is None:
            stats["malformed"] += 1
            continue

        class_name, _video_name = parsed
        if class_name not in class_mapping:
            stats["unknown_class"] += 1
            continue

        if line in entries:
            stats["duplicate"] += 1
        entries.add(line)
        class_counter[class_name] += 1

    return entries, class_counter, stats


def build_report(include_normal_class: bool) -> dict:
    klin_root = resolve_klin_root()
    splits_dir = klin_root / SPLITS_DIR_REL
    class_file = splits_dir / CLASS_IDS_FILE

    class_mapping = parse_class_mapping(
        class_file,
        include_normal_class=include_normal_class,
    )
    folds = discover_fold_split_files(splits_dir)

    fold_reports: list[dict] = []
    train_union: SplitEntries = set()
    test_union: SplitEntries = set()
    train_global_counts: Counter[str] = Counter()
    test_global_counts: Counter[str] = Counter()
    test_by_fold: dict[str, SplitEntries] = {}

    for fold_id, train_file, test_file in folds:
        train_entries, train_counts, train_stats = load_split_entries(
            train_file,
            class_mapping,
        )
        test_entries, test_counts, test_stats = load_split_entries(
            test_file,
            class_mapping,
        )

        fold_reports.append(
            {
                "fold_id": fold_id,
                "train_file": train_file.name,
                "test_file": test_file.name,
                "train_total": len(train_entries),
                "test_total": len(test_entries),
                "train_test_overlap": len(train_entries & test_entries),
                "train_class_counts": dict(sorted(train_counts.items())),
                "test_class_counts": dict(sorted(test_counts.items())),
                "train_parse_stats": train_stats,
                "test_parse_stats": test_stats,
            }
        )

        train_union |= train_entries
        test_union |= test_entries
        train_global_counts.update(train_counts)
        test_global_counts.update(test_counts)
        test_by_fold[fold_id] = test_entries

    pairwise_test_overlap: dict[str, int] = {}
    for fold_a, fold_b in combinations(sorted(test_by_fold), 2):
        key = f"{fold_a}-{fold_b}"
        pairwise_test_overlap[key] = len(test_by_fold[fold_a] & test_by_fold[fold_b])

    return {
        "dataset_root": str(klin_root / DATASET_DIR_NAME),
        "splits_dir": str(splits_dir),
        "include_normal_class": include_normal_class,
        "num_classes": len(class_mapping),
        "classes": dict(sorted(class_mapping.items(), key=lambda item: item[1])),
        "num_folds": len(folds),
        "fold_ids": [fold_id for fold_id, _train, _test in folds],
        "folds": fold_reports,
        "global": {
            "unique_train_entries": len(train_union),
            "unique_test_entries": len(test_union),
            "train_test_union_overlap": len(train_union & test_union),
            "train_class_counts_sum_over_folds": dict(
                sorted(train_global_counts.items())
            ),
            "test_class_counts_sum_over_folds": dict(
                sorted(test_global_counts.items())
            ),
            "pairwise_test_overlap": pairwise_test_overlap,
        },
    }


def main() -> None:
    args = parse_args()
    report = build_report(include_normal_class=args.include_normal)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Report saved to: {args.output}")
    print(f"Classes used: {report['num_classes']}")
    print(f"Folds found: {', '.join(report['fold_ids'])}")
    print("Per-fold totals:")
    for fold in report["folds"]:
        print(
            f"- fold {fold['fold_id']}: "
            f"train={fold['train_total']} "
            f"test={fold['test_total']} "
            f"overlap={fold['train_test_overlap']}"
        )


if __name__ == "__main__":
    main()

import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data_process.data_processing import group_setups, setup_pair_distances

N_FOLDS = 5
MIN_SAME = 2  # 같은 자리인 공통 약이 이 개수 이상이면 세팅 공유 (EDA 8.3)
RARE_MAX = 5  # bbox가 이 개수 이하인 약을 희귀 약으로 본다 (EDA 8.1)
# 분할 이름: <단위>_<라벨 버전>_s<seed>. 단위는 combo(B), setup<px>(C), setup<px>fix(C+)
SPLIT_PATTERN = re.compile(
    r"^(?:(?P<combo>combo)|setup(?P<dist>\d+)(?P<fix>fix)?)_(?P<label>\w+?)_s(?P<seed>\d+)$"
)
FOLD_COLS = ["file_name", "combo", "group", "fold", "split", "fixed"]


@dataclass(frozen=True)
class SplitSpec:
    """분할 이름에서 읽은 분할 설정. combo 분할도 세팅 묶음 누수를 재려고 same_dist를 갖는다."""

    name: str
    unit: str  # "combo" 또는 "setup"
    same_dist: int
    fix_rare: bool
    label_version: str
    seed: int


def parse_split_name(name: str, same_dist: int = 30) -> SplitSpec:
    """combo_v1_s42, setup30fix_v1_s42 같은 분할 이름을 SplitSpec으로 바꾼다.

    same_dist는 combo 분할에서 세팅 묶음 누수를 잴 때만 쓴다.
    """
    m = SPLIT_PATTERN.match(name)
    if m is None:
        raise ValueError(f"분할 이름 형식이 다릅니다: {name} (예: combo_v1_s42, setup30fix_v1_s42)")
    return SplitSpec(
        name=name,
        unit="combo" if m["combo"] else "setup",
        same_dist=same_dist if m["combo"] else int(m["dist"]),
        fix_rare=bool(m["fix"]),
        label_version=m["label"],
        seed=int(m["seed"]),
    )


def photo_units(labels: pd.DataFrame, same_dist: int, min_same: int = MIN_SAME) -> pd.DataFrame:
    """사진마다 조합(combo)과 세팅 묶음(setup)을 붙인 표를 만든다. 묶음은 박스 위치로 정한다."""
    units = pd.DataFrame({"file_name": sorted(labels["file_name"].unique())})
    units["combo"] = units["file_name"].str.split("_").str[0]
    _, setup = group_setups(
        setup_pair_distances(labels), sorted(units["combo"].unique()), same_dist, min_same
    )
    units["setup"] = units["combo"].map(setup)
    return units


def rare_classes(labels: pd.DataFrame, rare_max: int = RARE_MAX) -> list[int]:
    """bbox가 rare_max개 이하인 category_id 목록."""
    counts = labels["category_id"].value_counts()
    return sorted(int(c) for c in counts[counts <= rare_max].index)


def make_folds(labels: pd.DataFrame, spec: SplitSpec, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """bbox를 표본, category_id를 층, 분할 단위를 그룹으로 StratifiedGroupKFold를 만든다(EDA 8.4).

    fold × 사진마다 한 행이고 split은 train/val이다. fix_rare면 희귀 약이 든 세팅 묶음의 사진은
    모든 fold에서 train에 고정한다(fixed=True).
    """
    units = photo_units(labels, spec.same_dist)
    group_col = spec.unit
    boxes = labels[["file_name", "category_id"]].merge(units, on="file_name")
    fixed_groups = set()
    if spec.fix_rare:
        is_rare = boxes["category_id"].isin(rare_classes(labels))
        fixed_groups = set(boxes.loc[is_rare, group_col])
    units["group"] = units[group_col]
    units["fixed"] = units["group"].isin(fixed_groups)
    free = boxes[~boxes[group_col].isin(fixed_groups)].reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=spec.seed)
    with warnings.catch_warnings():
        # bbox가 n_folds개보다 적은 클래스가 있다는 경고. 희귀 약이 있어 당연히 나온다.
        warnings.simplefilter("ignore", UserWarning)
        folds = list(splitter.split(free, free["category_id"], free[group_col]))
    rows = []
    for fold, (_, val_idx) in enumerate(folds):
        val_files = set(free["file_name"].iloc[val_idx])
        rows.append(
            units.assign(
                fold=fold,
                split=units["file_name"].isin(val_files).map({True: "val", False: "train"}),
            )
        )
    return pd.concat(rows, ignore_index=True)[[*FOLD_COLS, "setup"]]


def fold_checks(folds: pd.DataFrame, labels: pd.DataFrame) -> list[dict]:
    """fold마다 검증 쪽 크기, 클래스 누락, 같은 조합·같은 묶음 누수를 잰다(EDA 8.4).

    누수는 같은 조합(묶음)의 사진이 train에 있는 val 사진의 비율이다.
    """
    checks = []
    for fold, rows in folds.groupby("fold"):
        val, train = rows[rows["split"] == "val"], rows[rows["split"] == "train"]
        val_labels = labels[labels["file_name"].isin(val["file_name"])]
        train_labels = labels[labels["file_name"].isin(train["file_name"])]
        train_cls, val_cls = set(train_labels["category_id"]), set(val_labels["category_id"])
        checks.append(
            {
                "fold": int(fold),
                "train_images": len(train),
                "val_images": len(val),
                "val_bboxes": len(val_labels),
                "val_only_classes": sorted(int(c) for c in val_cls - train_cls),
                "n_classes_not_in_val": len(train_cls - val_cls),
                "combo_leak": float(val["combo"].isin(set(train["combo"])).mean()),
                "setup_leak": float(val["setup"].isin(set(train["setup"])).mean()),
            }
        )
    return checks


def build_split(
    labels: pd.DataFrame, spec: SplitSpec, labels_sha256: str, n_folds: int = N_FOLDS
) -> tuple[pd.DataFrame, dict]:
    """분할을 만들고 누수를 검사해 (fold 표, meta)를 반환한다.

    같은 조합 누수가, setup 분할이면 같은 묶음 누수까지 0이 아니면 오류를 낸다.
    """
    folds = make_folds(labels, spec, n_folds)
    checks = fold_checks(folds, labels)
    for check in checks:
        if check["combo_leak"] > 0 or (spec.unit == "setup" and check["setup_leak"] > 0):
            raise ValueError(f"{spec.name} fold {check['fold']}에 누수가 있습니다: {check}")
    fixed = folds[(folds["fold"] == 0) & folds["fixed"]]
    meta = {
        **asdict(spec),
        "n_folds": n_folds,
        "min_same": MIN_SAME,
        "rare_max": RARE_MAX,
        "rare_classes": rare_classes(labels),
        "labels_sha256": labels_sha256,
        "n_images": int(folds["file_name"].nunique()),
        "fixed_train_images": len(fixed),
        "fixed_train_groups": int(fixed["group"].nunique()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checks": checks,
    }
    return folds, meta


def save_split(folds: pd.DataFrame, out_dir: Path) -> list[Path]:
    """fold마다 fold_k.csv(사진마다 한 행)를 저장하고 경로를 반환한다. 옛 fold 파일은 지운다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("fold_*.csv"):
        old.unlink()
    paths = []
    for fold, rows in folds.groupby("fold"):
        path = out_dir / f"fold_{fold}.csv"
        rows[FOLD_COLS].to_csv(path, index=False, encoding="utf-8")
        paths.append(path)
    return paths

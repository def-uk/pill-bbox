from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from data_process.data_yaml import data_yaml_path
from models.evaluate import map75_95
from src.config import RUNS_DIR, get_device, init_wandb

MAIN_METRIC_KEY = "metrics/mAP75-95(B)"  # results.csv·wandb에 남는 열 이름
# ultralytics가 받는 옵티마이저. auto는 반복 수(⌈학습 사진 수/64⌉ × epoch)가 10000을 넘으면
# MuSGD, 아니면 AdamW를 고르고 lr0를 직접 계산해 lr0 인자를 무시한다.
# 우리 데이터(100 epoch에 300–400회)에서는 항상 AdamW, lr0 ≈ 0.00017이 된다.
OPTIMIZERS = ["auto", "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"]
# lr0를 정하지 않았을 때의 시작값 (ultralytics 설명: SGD=1e-2, Adam/AdamW=1e-3)
SGD_LR0 = 0.01
ADAM_LR0 = 0.001
# 증강 설정 묶음. 같은 분할·fold에서 on과 off를 학습해 증강 효과를 비교한다.
# on: 두 모델에 같게 쓰는 기본값 (계획 [5], EDA 9.3). 적지 않은 인자는 ultralytics 기본값이다
#     (hsv_s 0.7, hsv_v 0.4, translate 0.1, scale 0.5, mosaic 1.0, close_mosaic 10).
# - degrees: 알약은 방향과 무관. 회전하면 박스가 외접 사각형이라 느슨해지므로 작게 시작한다.
# - fliplr·flipud: 각인이 거울상이 되므로 끈다.
# - hsv_h: 색이 약 구별 단서라 색상 변화를 줄인다.
# - copy_paste: segmentation 라벨이 없어 ultralytics 구현이 동작하지 않는다.
# - perspective: 각도별 bbox 차이가 1–2%라 쓰지 않는다.
# off: 학습 이미지를 바꾸는 증강을 모두 끈다. 크기 맞춤(letterbox)만 남는다.
AUG_PRESETS = {
    "on": {
        "degrees": 10.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.005,
        "copy_paste": 0.0,
        "perspective": 0.0,
    },
    "off": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    },
}


class Map7595Trainer(DetectionTrainer):
    """매 epoch 검증의 mAP75-95를 fitness로 써서 best.pt와 early stopping을 정하는 trainer.

    ultralytics 기본 fitness는 mAP50-95다. 검증·저장 흐름은 그대로 두고 fitness 값만 바꾼다.
    """

    def validate(self) -> tuple[dict | None, float | None]:
        best = self.best_fitness
        metrics, fitness = super().validate()
        if metrics is None:
            return metrics, fitness
        fitness = map75_95(self.validator.metrics.box)
        metrics[MAIN_METRIC_KEY] = round(fitness, 5)
        self.best_fitness = fitness if best is None or best < fitness else best
        return metrics, fitness


def default_lr0(optimizer: str) -> float | None:
    """옵티마이저별 기본 lr0. auto는 ultralytics가 정하므로 None."""
    if optimizer == "auto":
        return None
    return SGD_LR0 if "SGD" in optimizer else ADAM_LR0


def run_dir(
    model_name: str,
    split_name: str,
    fold: int,
    exp: str,
    aug: str,
    optimizer: str = "auto",
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """학습 결과 폴더 <runs_dir>/<exp>/<모델>_<분할>_aug<on|off>_opt<옵티마이저>_fold<k>를 반환."""
    stem = Path(model_name).stem
    return Path(runs_dir) / exp / f"{stem}_{split_name}_aug{aug}_opt{optimizer}_fold{fold}"


def is_finished(last: Path) -> bool:
    """last.pt가 끝까지 학습된 체크포인트인지 본다. 학습을 마치면 epoch가 -1이 된다."""
    return torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1) == -1


def train(
    model_name: str,
    label_version: str,
    split_name: str,
    fold: int,
    epochs: int,
    exp: str,
    imgsz: int = 640,
    batch: int = 16,
    seed: int = 42,
    fraction: float = 1.0,
    aug: str = "on",
    optimizer: str = "auto",
    lr0: float | None = None,
    cos_lr: bool = False,
    runs_dir: Path = RUNS_DIR,
    resume: bool = False,
    **train_kwargs: object,
) -> Path:
    """data_main이 만든 fold의 data.yaml로 YOLO를 학습하고 결과 폴더를 반환한다.

    model_name은 ultralytics 가중치 이름(yolo11m.pt, yolo26m.pt 등)이다. aug는 AUG_PRESETS의
    이름(on/off)이고 결과 폴더 이름에 들어간다. train_kwargs는 증강 설정을 덮어쓰거나 다른 학습
    인자를 더한다. best.pt는 검증 mAP75-95가 가장 높은 epoch다.
    optimizer는 OPTIMIZERS 중 하나이고 결과 폴더 이름에 들어간다. lr0를 생략하면 SGD 계열 0.01,
    그 밖은 0.001로 시작한다. auto는 lr0를 무시하므로 함께 줄 수 없다. cos_lr은 cosine 학습률 감소.
    결과는 <runs_dir>/<exp>/<이름>/에 쌓인다. Colab에서는 runs_dir을 Drive 아래로 두면 런타임이
    끊겨도 남는다. resume=True면 그 폴더의 last.pt에서 이어 학습하고, 이미 끝난 학습은 건너뛴다.
    라벨 버전·분할·fold·seed는 wandb config에 남긴다. wandb를 끄려면 WANDB_MODE=disabled.
    """
    data = data_yaml_path(label_version, split_name, fold)
    if not data.is_file():
        raise FileNotFoundError(
            f"{data}가 없습니다. 먼저 data_process.data_main --label {label_version} "
            f"--split {split_name}을 실행하세요."
        )
    if aug not in AUG_PRESETS:
        raise ValueError(f"증강 설정 {aug}이 없습니다: {list(AUG_PRESETS)}")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"옵티마이저 {optimizer}이 없습니다: {OPTIMIZERS}")
    if optimizer == "auto" and lr0 is not None:
        raise ValueError("optimizer=auto는 lr0를 무시합니다. lr0를 정하려면 옵티마이저를 고르세요.")
    lr0 = default_lr0(optimizer) if lr0 is None else lr0
    save_dir = run_dir(model_name, split_name, fold, exp, aug, optimizer, runs_dir)
    run_name = save_dir.name
    last = save_dir / "weights" / "last.pt"
    if last.is_file() and not resume:
        raise FileExistsError(
            f"{save_dir}에 이전 학습이 있습니다. "
            "이어서 하려면 --resume, 새로 하려면 다른 --exp를 쓰세요."
        )
    if resume and last.is_file() and is_finished(last):
        print(f"이미 끝난 학습이라 건너뜁니다: {save_dir}")
        return save_dir
    args = {
        **AUG_PRESETS[aug],
        **train_kwargs,
        "data": str(data),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "deterministic": True,
        "fraction": fraction,
        "optimizer": optimizer,
        **({} if lr0 is None else {"lr0": lr0}),
        "cos_lr": cos_lr,
        "device": get_device(),
        "project": str(save_dir.parent),
        "name": run_name,
        "exist_ok": True,  # 폴더 이름을 고정해 이어서 학습·평가가 같은 경로를 찾게 한다
    }
    config = {
        "model": model_name,
        "label_version": label_version,
        "split_name": split_name,
        "fold": fold,
        "aug": aug,
        **{k: v for k, v in args.items() if k not in ("data", "project", "name")},
    }
    if resume and last.is_file():
        # 이어 학습은 체크포인트의 학습 인자를 그대로 쓰고, 이 PC의 경로·장치만 바꾼다.
        model = YOLO(last)
        args = {
            "resume": True,
            "data": args["data"],
            "device": args["device"],
            "save_dir": str(save_dir),
        }
        print(f"{last}에서 이어서 학습합니다.")
    else:
        model = YOLO(model_name)
    run = init_wandb(model, run_name=f"{exp}/{run_name}", config={**config, "resume": resume})
    # auto가 실제로 고른 옵티마이저와 lr0는 학습 시작 때 정해지므로 그때 wandb config에 더한다.
    model.add_callback(
        "on_train_start",
        lambda trainer: run.config.update(
            {"optimizer_used": trainer.args.optimizer, "lr0_used": trainer.args.lr0},
            allow_val_change=True,
        ),
    )
    try:
        model.train(trainer=Map7595Trainer, **args)
    finally:
        run.finish()
    return Path(model.trainer.save_dir)

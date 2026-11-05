"""fold별 분할·설정을 대조하고 중단된 YOLO 학습을 이어서 실행한다."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from ultralytics import YOLO

from models.validation_train import HoldoutTrainer
from src.config import init_wandb


def _write_json(path: Path, value: dict) -> None:
    """JSON을 임시 파일에 기록한 뒤 교체한다."""
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def train_fold(
    model_source: str,
    train_args: dict,
    split_rows: pd.DataFrame,
    manifest: dict,
    *,
    resume: bool = True,
    resume_checkpoint: Path | None = None,
    callbacks: dict[str, list[Callable[[Any], None]]] | None = None,
    wandb_enabled: bool = False,
    data_fingerprint: str | None = None,
) -> Path:
    """완료 fold는 재사용하고 미완료 fold는 last.pt의 다음 epoch부터 학습한다."""
    context = {
        "fold": manifest["fold"],
        "model_source": str(model_source),
        "train_args": {key: value for key, value in train_args.items() if key != "device"},
        "split_sha256": hashlib.sha256(split_rows.to_csv(index=False).encode()).hexdigest(),
        "data_yaml_sha256": hashlib.sha256(Path(train_args["data"]).read_bytes()).hexdigest(),
        "data_fingerprint": data_fingerprint,
    }
    checkpoint = None
    if resume_checkpoint is not None:
        if not resume:
            raise ValueError("재개 경로를 사용할 때는 RESUME_TRAINING=True여야 합니다.")
        checkpoint = Path(resume_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    elif resume:
        pattern = re.compile(re.escape(train_args["name"]) + r"(?:-\d+)?")
        candidates = sorted(
            path
            for path in Path(train_args["project"]).glob("*/weights/last.pt")
            if pattern.fullmatch(path.parent.parent.name)
        )
        if len(candidates) > 1:
            raise ValueError(
                "재개 후보가 여러 개입니다. RESUME_CHECKPOINTS에 fold별 경로를 지정하세요:\n"
                + "\n".join(map(str, candidates))
            )
        checkpoint = candidates[0] if candidates else None

    model = None
    if checkpoint is not None:
        run_dir = checkpoint.parent.parent
        context_path = run_dir / "fold_training.json"
        if not context_path.is_file() or json.loads(context_path.read_text()) != context:
            raise ValueError(
                f"fold {manifest['fold']}: 저장 설정·분할·데이터가 현재 실행과 다릅니다. "
                "기존 설정을 복원하거나 RESUME_TRAINING=False로 새 실험을 시작하세요."
            )
        best = checkpoint.parent / "best.pt"
        if (run_dir / "training_complete.json").is_file() and best.is_file():
            print(f"fold {manifest['fold']}: 완료된 학습 재사용 {best}", flush=True)
            return best
        model = YOLO(checkpoint)
        state = model.ckpt
        completed = int(state.get("epoch", -1)) + 1
        if best.is_file() and (
            completed >= train_args["epochs"]
            or (state.get("epoch") == -1 and state.get("optimizer") is None)
        ):
            _write_json(run_dir / "training_complete.json", {"best": str(best)})
            print(f"fold {manifest['fold']}: 종료된 학습 재사용 {best}", flush=True)
            return best
        if completed <= 0 or state.get("optimizer") is None:
            raise ValueError(f"학습 상태가 없는 모델은 재개할 수 없습니다: {checkpoint}")
        print(
            f"fold {manifest['fold']}: {completed} epoch 완료, "
            f"{completed + 1} epoch부터 재개합니다: {checkpoint}",
            flush=True,
        )
    if model is None:
        model = YOLO(model_source)

    def save_context(trainer: HoldoutTrainer) -> None:
        """첫 epoch 전에 분할·설정을 저장해 프로세스가 끊겨도 다시 찾을 수 있게 한다."""
        run_dir = Path(trainer.save_dir)
        _write_json(run_dir / "fold_training.json", context)
        _write_json(run_dir / "holdout_manifest.json", manifest)
        temporary = run_dir / "holdout_split.pending.csv"
        split_rows.to_csv(temporary, index=False)
        temporary.replace(run_dir / "holdout_split.csv")

    model.add_callback("on_pretrain_routine_end", save_context)
    for event, functions in (callbacks or {}).items():
        for function in functions:
            model.add_callback(event, function)
    run = (
        init_wandb(model, run_name=train_args["name"], config=train_args) if wandb_enabled else None
    )
    try:
        model.train(trainer=HoldoutTrainer, **train_args, resume=checkpoint is not None)
    finally:
        if run is not None:
            run.finish()
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    _write_json(best.parent.parent / "training_complete.json", {"best": str(best)})
    return best

import os
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.utils.callbacks.wb import callbacks as wandb_callbacks

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env의 WANDB_API_KEY 등을 환경 변수로 불러온다. 이미 설정된 환경 변수가 우선한다.
load_dotenv(PROJECT_ROOT / ".env")

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "pillbbox")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None  # 팀 이름. 비워 두면 개인 계정에 기록


def get_device() -> str:
    """Return the best available device: CUDA (Windows/Linux) > MPS (Mac) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_wandb(
    model: YOLO | None = None,
    run_name: str | None = None,
    config: dict | None = None,
    **kwargs,
) -> wandb.Run:
    """wandb run을 시작한다. model을 넘기면 그 모델의 학습 로그가 이 run에 기록된다.

    ultralytics 전역 설정(settings wandb=True)은 바꾸지 않고 넘긴 모델에만 callback을 붙이므로,
    이 프로젝트 밖의 YOLO 학습에는 영향이 없다.
    끄고 싶을 때는 WANDB_MODE=disabled, 인터넷 없이 기록만 남길 때는 WANDB_MODE=offline.
    """
    if model is not None:
        for event, callback in wandb_callbacks.items():
            model.add_callback(event, callback)
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
        dir=PROJECT_ROOT,
        **kwargs,
    )

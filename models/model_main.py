"""모델 학습 진입점. data_process.data_main으로 data.yaml을 만든 뒤 실행한다.

fold마다 학습이 끝나면 best.pt(검증 mAP75-95 최고 epoch)로 평가하고, 마지막에 실험 요약을 만든다.

    # 동작 확인 (짧게)
    WANDB_MODE=disabled uv run python -m models.model_main --model yolo11m.pt yolo26m.pt \\
        --split combo_v1_s42 --folds 0 --epochs 1 --fraction 0.1 --aug on off --exp smoke

    # Colab: 결과를 Drive에 두고, 끊기면 같은 명령에 --resume을 붙여 다시 실행
    uv run python -m models.model_main --model yolo26m.pt --split setup30fix_v1_s42 \\
        --epochs 100 --exp base_v1 --runs-dir /content/drive/MyDrive/pill-bbox/runs --resume
"""

import argparse
from pathlib import Path

from models.evaluate import MAIN_METRIC, evaluate_fold, results_dir, summarize_exp
from models.train import AUG_PRESETS, train
from src.config import RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 × fold 학습과 평가")
    parser.add_argument("--model", nargs="+", default=["yolo11m.pt"], help="가중치 이름")
    parser.add_argument("--label", default="v1", help="라벨 버전")
    parser.add_argument("--split", required=True, help="분할 이름 (예: combo_v1_s42)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--exp", required=True, help="결과 폴더 <runs-dir>/<exp>와 wandb run 이름")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0, help="학습 사진 중 쓸 비율")
    parser.add_argument(
        "--aug", nargs="+", default=["on"], choices=list(AUG_PRESETS), help="증강 켜기/끄기"
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=RUNS_DIR, help="결과 폴더. Colab에서는 Drive 아래 경로"
    )
    parser.add_argument(
        "--resume", action="store_true", help="last.pt에서 이어 학습하고 끝난 학습은 건너뜀"
    )
    args = parser.parse_args()
    for model_name in args.model:
        for aug in args.aug:
            for fold in args.folds:
                save_dir = train(
                    model_name,
                    args.label,
                    args.split,
                    fold,
                    epochs=args.epochs,
                    exp=args.exp,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    seed=args.seed,
                    fraction=args.fraction,
                    aug=aug,
                    runs_dir=args.runs_dir,
                    resume=args.resume,
                )
                _, summary = evaluate_fold(
                    save_dir / "weights" / "best.pt",
                    args.label,
                    args.split,
                    fold,
                    exp=args.exp,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    runs_dir=args.runs_dir,
                )
                print(f"{save_dir.name}: {MAIN_METRIC} {summary[MAIN_METRIC]:.4f}")
    summary = summarize_exp(args.exp, args.runs_dir)
    print(summary[[("n_folds", "count"), (MAIN_METRIC, "mean"), (MAIN_METRIC, "std")]])
    print(f"요약: {results_dir(args.exp, args.runs_dir) / 'summary.csv'}")


if __name__ == "__main__":
    main()

"""모델 학습 진입점. data_process.data_main으로 data.yaml을 만든 뒤 실행한다.

    # 동작 확인 (짧게)
    WANDB_MODE=disabled uv run python -m models.model_main --model yolo11m.pt yolo26m.pt \\
        --split combo_v1_s42 --folds 0 --epochs 1 --fraction 0.1 --aug on off --exp smoke
"""

import argparse

from models.train import AUG_PRESETS, train


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 × fold 학습")
    parser.add_argument("--model", nargs="+", default=["yolo11m.pt"], help="가중치 이름")
    parser.add_argument("--label", default="v1", help="라벨 버전")
    parser.add_argument("--split", required=True, help="분할 이름 (예: combo_v1_s42)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--exp", required=True, help="결과 폴더 runs/<exp>와 wandb run 이름 앞부분")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0, help="학습 사진 중 쓸 비율")
    parser.add_argument(
        "--aug", nargs="+", default=["on"], choices=list(AUG_PRESETS), help="증강 켜기/끄기"
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
                )
                print(f"{model_name} aug {aug} fold {fold}: {save_dir}")


if __name__ == "__main__":
    main()

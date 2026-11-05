# 데이터 파이프라인 사용법

원본 JSON에서 YOLO 학습 입력(`data.yaml`)까지 만들고, 그 데이터로 학습·평가·test 예측을 하는 방법입니다.
모든 팀원과 에이전트가 같은 라벨, 같은 분할로 실험하도록 데이터는 이 파이프라인으로만 만듭니다.

## 한 번에 실행하기

```bash
uv sync
uv run python -m data_process.data_main --label v1          # 0–4단계: 원본 → 라벨 → 분할 → YOLO 데이터
```

`data/raw/`가 비어 있으면 `.env`의 `DATA_URL`로 원본을 먼저 받습니다. 전체 25초 정도 걸리고, 실행할 때마다 같은 결과를 다시 만듭니다.
끝나면 학습에 넘길 `data.yaml`이 fold마다 생깁니다.

```text
data/processed/yolo/v1/splits/combo_v1_s42/fold_0/data.yaml
...
data/processed/yolo/v1/splits/setup30fix_v1_s42/fold_4/data.yaml
```

`data.yaml`과 `train.txt`에는 이 PC의 절대 경로가 들어갑니다. **다른 PC나 Colab에서는 폴더를 복사하지 말고 그곳에서 다시 실행**합니다.

## 단계와 검사

| 단계 | 하는 일 | 자동 검사 (어기면 멈추거나 경고) |
|---|---|---|
| [0] 원본 | 없으면 받기 | train 232장, JSON 763개, test 842장 |
| [1] 라벨 로드 | JSON을 bbox 한 행 표로 읽기 | JSON의 width·height와 실제 이미지 크기 비교 |
| [2] 라벨 버전 | 보정 기록을 적용해 `labels_<버전>.csv` 저장 | 이미지 밖 좌표, 크기 0 이하, 같은 사진·같은 약 중복이 0 (raw 제외) |
| [3] 분할 | 5 fold를 만들어 `fold_k.csv` 저장 | fold마다 같은 조합 누수 0%, 세팅 묶음 분할은 묶음 누수 0% |
| [4] YOLO 변환 | 이미지 링크, 라벨 txt, `class_map.csv`, fold별 `data.yaml` | 좌표 0–1, 클래스 0–55, 라벨 수 일치, txt → 픽셀 역변환 오차, 무작위 5장 그림 |

검사 결과는 `labels_<버전>.json`, `splits/<분할>/meta.json`, `yolo/<버전>/check.json`에 남습니다.

## 라벨 버전

| 버전 | 내용 | 용도 |
|---|---|---|
| `raw` | 원본 그대로 (763개, 이미지 밖 박스 1개는 YOLO 변환에서 빠짐) | 보정 효과를 재는 비교 기준 |
| `v1` | 라벨 누락 8개 추가, 위치 수정 5개(kwon), 19232 잘린 캡슐 확대 (771개) | **기본 학습 라벨** |
| `v2` | v1 + 여백 검수표에서 **승인한** 박스만 알약 몸체에 맞게 줄임 | 여백 기준 실험. 승인이 0건이면 v1과 같음 |
| `v2auto` | v1 + 여백 검수표의 후보를 **검수 없이** 모두 줄임(`거부`한 행만 뺌). 지금 751개, 면적 평균 0.88배 | 여백 축소 효과를 미리 보는 실험용. 기본 라벨로 쓰지 않음 |

`v2auto`를 v1과 비교할 때는 두 가지를 지킵니다.

- 분할은 `combo_v2auto_s42`를 씁니다. fold 구성이 `combo_v1_s42`와 같습니다. `setup30fix`는 박스 위치로 묶음을 정하므로 fold마다 23–78장이 v1과 다릅니다.
- v2auto로 학습한 `best.pt`도 **v1 라벨로 평가**합니다(`data_yaml_path("v1", "combo_v1_s42", k)`의 val로 검증). 줄인 박스로 잰 검증 점수는 여백이 있는 정답 기준의 점수와 비교할 수 없습니다.

보정은 원본 JSON을 고치지 않고 `data/processed/corrections.csv`(보정 기록)로 적용합니다. 기록마다 "지금 박스가 기록의 이전 좌표와 같을 때만" 적용되므로 잘못된 기록이 조용히 섞이지 않습니다.

## 분할 이름

`<단위>_<라벨 버전>_s<seed>` 형식입니다. 라벨 버전이 다르면 세팅 묶음이 달라지므로 분할도 따로 만듭니다.

| 이름 | 단위 | 특징 |
|---|---|---|
| `combo_v1_s42` | 조합 (같은 약 조합을 각도만 바꿔 찍은 사진은 한쪽에) | 세팅을 공유하는 다른 조합 누수가 fold마다 48–75% 남음 |
| `setup30fix_v1_s42` | 세팅 묶음 (30px 기준) + 희귀 약 묶음은 항상 train | 누수 0%. 검증 사진이 26–43장으로 적음 |
| `setup30_v1_s42` | 세팅 묶음, 희귀 약 고정 없음 | 만들 수는 있지만 기본 실행에서는 만들지 않음 |

`--split`을 생략하면 `combo_<버전>_s42`와 `setup30fix_<버전>_s42` 두 개를 만듭니다. 한 fold 점수로 모델을 고르지 말고 5 fold 평균·표준편차로 비교합니다.

## 학습

학습은 최종 모델 노트북 `notebooks/yolo11m_100_aihub_not_background.ipynb`에서 합니다. 노트북이 이 문서의 데이터 파이프라인(`data_process.data_main`)으로 라벨·분할·YOLO 입력을 만든 뒤 Ultralytics로 직접 학습합니다. 설정은 노트북의 "기본 설정 한눈에" 표에 있습니다.

- 모델은 `yolo11m.pt`, 100 epoch(patience 10), MuSGD lr0 0.001, 배치 16, 이미지 640, seed 42입니다.
- **전체 파인튜닝**입니다. 분류 출력층만 새로 만들고 나머지 층은 사전학습 가중치에서 시작해 모두 학습합니다(`freeze` 없음).
- `best.pt`는 검증 **mAP75-95**가 가장 높은 epoch입니다. early stopping도 이 값을 씁니다.
- 장치는 `get_device()`가 고릅니다(cuda → mps → cpu). Mac(MPS)은 느리므로 **본 학습은 Colab GPU**에서 합니다.

## 평가와 test 예측

주 지표는 **mAP75-95**입니다. IoU 0.75, 0.80, 0.85, 0.90, 0.95 다섯 기준의 AP 평균으로, 박스가 알약에 꼭 맞아야 점수가 납니다. 학습 중 `best.pt` 선택과 early stopping이 이 값을 씁니다. mAP50, mAP75, mAP50-95도 함께 기록합니다. 대회 채점 지표 mAP@[0.75:0.95]와 같습니다.

fold 평가는 노트북이 학습 뒤에 직접 합니다. 학습한 `best.pt`로 test를 예측하고 제출 csv를 따로 만들 때:

```python
from models.evaluate import make_submission, predict_test

weights = "runs/base_v1/yolo11m_setup30fix_v1_s42_augon_optauto_fold0/weights/best.pt"
predict_test(weights, "v1", exp="base_v1")  # test 842장 예측
make_submission(weights, "v1", exp="base_v1")  # test 예측 → 제출 csv
```

- 결과는 `runs/<exp>/results/`에 저장됩니다. Drive에 학습했다면 `runs_dir=`에 같은 Drive 경로를 넘깁니다.
- `predict_test`: `category_id`와 픽셀 좌표 `[x, y, w, h]`로 되돌려 `<run 이름>_test_pred.csv`로 저장합니다.
- `make_submission`: test를 예측해 제출 형식 `annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score`로 `<run 이름>_submission_conf<conf>.csv`에 저장합니다. `image_id`는 test 파일 이름의 숫자, `category_id`는 원본 JSON의 id입니다. 기본 `conf`는 0.001입니다. mAP 채점이라 점수 낮은 박스를 더해도 점수가 떨어지지 않기 때문입니다(파일 약 20MB). 터미널에서는 `uv run python -m models.evaluate --weights <best.pt> --exp <exp>`.

## 여백 검수 (v2 라벨)

정답 박스는 알약보다 사방 8px 정도 넉넉합니다. v2는 이 여백을 줄인 라벨이며, **사람이 승인한 박스만** 바꿉니다.

```bash
uv run python -m data_process.data_main --prepare-padding-review   # 측정 + 검수표 + 그림 (약 3분)
```

1. `data/processed/padding/review_pages/page_01.png`부터 봅니다. 빨강은 지금 박스, 초록 점선은 제안 박스입니다. 제목은 `검수 번호 약 코드 여백 비율`입니다.
2. `data/processed/padding_review.csv`의 `decision` 칸에 `승인` 또는 `거부`를 적고 `reviewer`, `comment`를 채웁니다. 빈칸은 미검수입니다.
3. `uv run python -m data_process.data_main --label v2`로 v2를 만듭니다. 승인·거부·미검수 건수는 `labels_v2.json`에 남습니다.

- 후보는 GrabCut 여백 비율 10% 이상인 박스입니다(`make_labels.py`의 `PADDING_MIN`). v1 기준 751개입니다.
- 자동 추정이 불안정한 6개는 후보에서 빠지고 원래 박스를 씁니다(`padding/needs_manual_v1.csv`).
- 측정을 다시 돌려도 박스 값이 같은 행의 검수 결과는 유지됩니다.

### 검수 현황 (2026-10-27)

| 대상 | 상태 |
|---|---|
| 19232 잘린 캡슐 확대 `[523, 201, 232, 241]` → `[527, 211, 224, 323]` | 그림으로 확인해 v1에 적용. **팀 확인 필요** (`make_labels.py`의 `CUT_CAPSULE_REVIEWED`) |
| 여백 보정 후보 751개 | **0 / 751 검수**. 지금 v2는 v1과 같음 |
| 여백 자동 추정이 불안정한 6개 | 후보에서 제외, 원래 박스 유지 |

## 산출물 위치

```text
data/processed/
├── added_annotations.csv, fixed_annotations.csv   # kwon 보정 원 기록 (입력)
├── corrections.csv                                 # 통합 보정 기록 (자동 생성)
├── padding_review.csv                              # 여백 검수표 (사람이 decision을 채움)
├── raw_check.json                                  # 원본 수·해상도 검사
├── labels/labels_<버전>.csv, .json                 # 라벨 버전과 검사·적용 기록
├── splits/<분할>/fold_k.csv, meta.json            # 사진별 train/val, 누수 검사
├── yolo/<버전>/                                    # images/, labels/, class_map.csv, splits/<분할>/fold_k/data.yaml
└── padding/                                        # 여백 측정 결과와 검수 그림

runs/<exp>/                                         # 커밋 안 함. --runs-dir로 위치 변경
├── <run 이름>/                                     # weights/best.pt, last.pt, results.csv
└── results/                                        # fold별 평가 CSV, summary.csv, test 예측
```

- `data/processed/` 바로 아래의 csv만 Git에 올릴 수 있습니다(`.gitignore`). 하위 폴더는 모두 각자 다시 만듭니다.
- 사람이 손으로 고치는 파일은 `padding_review.csv`의 `decision`, `reviewer`, `comment` 칸뿐입니다. 나머지는 직접 고치지 말고 코드나 보정 기록을 바꾼 뒤 다시 실행합니다.

## 코드 위치

| 파일 | 역할 |
|---|---|
| `data_process/data_main.py` | 0–4단계 진입점, 여백 검수 준비 |
| `data_process/make_labels.py` | 보정 기록, 라벨 버전(`LABEL_VERSIONS`), 여백 검수표, YOLO txt 변환 |
| `data_process/split.py` | 분할 이름 해석, fold 생성, 누수 검사 |
| `data_process/data_yaml.py` | `class_map.csv`, fold별 `train.txt`·`val.txt`·`data.yaml` |
| `data_process/move_files.py` | 이미지 링크(안 되면 복사) |
| `notebooks/yolo11m_100_aihub_not_background.ipynb` | 최종 모델 학습(mAP75-95로 best.pt 선택), 평가, 제출 csv |
| `models/evaluate.py` | test 예측, 제출 csv |

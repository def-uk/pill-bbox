# 실험 지도

main에는 최종 모델 노트북 [`notebooks/yolo11m.ipynb`](../notebooks/yolo11m.ipynb)와 EDA 노트북만 있습니다. 나머지 실험 노트북은 **모델별 실험 브랜치**에 있습니다. 실험은 하나씩 PR로 들어갔고, 설정과 결과는 각 PR 본문에 적혀 있습니다.

| 브랜치 | 내용 |
|---|---|
| [`exp/yolo11m`](https://github.com/def-uk/pill-bbox/tree/exp/yolo11m/notebooks) | YOLO11m 베이스라인부터 최종 설정까지, 증강·OCR 실험 |
| [`exp/yolo26m`](https://github.com/def-uk/pill-bbox/tree/exp/yolo26m/notebooks) | YOLO26m 베이스라인, NMS-free, 외부 데이터, copy-paste, OCR |
| [`exp/faster_r-cnn`](https://github.com/def-uk/pill-bbox/tree/exp/faster_r-cnn/notebooks) | Faster R-CNN 베이스라인, 모자이크, copy-paste, OCR |

읽는 법:

- **Kaggle**은 대회 점수 mAP@[0.75:0.95]입니다. 모두 seed 하나로 한 번씩 돌린 값이라 ±0.01 차이는 노이즈와 구분되지 않습니다.
- **A안**은 대회 밖 알약(외부 전용 클래스)을 배경으로 두거나 제출에서 뺀 것이고, **B안**은 새 클래스로 학습해 약 코드로 제출한 것입니다. A안과 B안 점수는 서로 비교하지 않습니다.
- PR 본문에 Kaggle 점수가 없는 실험은 "–"로 두었습니다. 검증 점수는 PR에 있습니다.

## YOLO11m (`exp/yolo11m`)

| 단계 | 실험 | 노트북 | PR | Kaggle |
|---|---|---|---|---|
| 베이스라인 | 각자 베이스라인(30 에폭) | `yolo11m_kwon.ipynb`, `yolo11m_kang.ipynb` | #98, #100 | – |
| 베이스라인 | 두 베이스라인 병합(30 에폭) | `yolo11m.ipynb` | #102 | 0.29946 |
| 학습량 | 100 에폭·patience 10 | `yolo11m_epoch100.ipynb` | #103 | 0.41485 |
| 증강 | 대회 사진만, 모자이크 끔 | `yolo11m_mosaic.ipynb` (main에 들어갔다가 #144에서 정리, 이력에만 있음) | #106 | 약 0.36 |
| 외부 데이터 | AI Hub 추가 (A안) | `yolo11m_epoch100_aihub.ipynb` | #105 | 0.44885 |
| 증강 | 위 + 모자이크 끔 (A안) | `yolo11m_100_aihub_mosaic.ipynb` | #107 | 0.46281 |
| 외부 데이터 | **외부 전용 알약을 새 클래스로 학습 (B안, 최종 설정)** | `yolo11m_100_aihub_not_background.ipynb` | #109 | **0.63060** (fold 1) |
| OCR | 각인 OCR 결합 | `yolo11m-ocr_kaggle_nature.ipynb` | #117 | – |
| 증강 | 채도·색조·밝기 줄이기 | `yolo11m_saturation.ipynb` | #121 | – |
| OCR | 각인 OCR v2 | `yolo11m_ocr_v2.ipynb` | #122 | – |
| 증강 | 최종 설정에서 모자이크 끔 | `yolo11m_best_off_mosaic(kaggle).ipynb` | #123 | – |
| 오버샘플링 | 전체 클래스 오버샘플링, 50 에폭 | `yolo11m_over_all.ipynb`, `yolo11m_best_over.ipynb` | #132 | 0.61619 |
| 증강 | 모자이크·회전 끄고 색 증강 낮춤 | `yolo11m_ocr_v5.ipynb` | #136 | – |
| 최종 | fold 1 `best.pt`를 최종 모델로 정리 | main의 `yolo11m.ipynb` | #137, #142, #149 | 0.63060 |

최종 설정 위에 한 가지씩 더한 실험(#121–#123, #132, #136)은 모두 0.63060을 넘지 못했습니다(보고서 5.3).

## YOLO26m (`exp/yolo26m`)

| 단계 | 실험 | 노트북 | PR | Kaggle |
|---|---|---|---|---|
| 베이스라인 | 30 에폭, NMS 헤드로 평가 | `exp_yolo26m.ipynb` | #101 | 0.17877 |
| 실행 환경 | Mac(MPS)에서도 실행 | `exp_yolo26m.ipynb` | #110 | – |
| 평가 헤드 | NMS-free(one2one) 헤드로 평가 | `exp_yolo26m_nms_free.ipynb` | #111 | 0.18437 |
| 외부 데이터 | AI Hub 추가 (A안) | `exp_yolo26m_nms_free_aihub.ipynb` | #112 | 0.45379 |
| 증강 | 위 + 모자이크 끔 (A안) | `exp_yolo26m_nms_free_aihub_mosaic0.ipynb` | #113 | 0.44731 |
| 증강 | 위 + bbox copy-paste (A안 / B안) | `exp_yolo26m_nms_free_aihub_copypaste.ipynb` | #114 | 0.46415 / **0.60721** |
| 외부 데이터 | 2차 EDA로 선별·검수한 AI Hub (B안) | `exp_yolo26m_nms_free_aihub_copypaste_eda.ipynb` | #116 | 0.60008 |
| OCR | 각인 OCR 재분류 (B안) | `exp_yolo26m_nms_free_aihub_copypaste_eda_ocr.ipynb` | #120 | 0.58768 |

베이스라인 0.18로 한때 후보에서 뺐다가 다시 넣은 과정은 [YOLO26m 제외 결정 재검토](YOLO26M_REVIEW.md)에 정리했습니다.

## Faster R-CNN (`exp/faster_r-cnn`)

| 단계 | 실험 | 노트북 | PR | Kaggle |
|---|---|---|---|---|
| 베이스라인 | 30 에폭 | `faster_rcnn.ipynb` | – | 0.25918 |
| 증강 | 모자이크 | `faster_rcnn_mosaic.ipynb` | #108 | – |
| 증강 | bbox copy-paste | `faster_rcnn_jang*.ipynb` | #118(닫힘), #119 | – |
| OCR | 외부 데이터 + 각인 OCR | `faster_rcnn_aihub_ocr.ipynb` | – | – |
| 최종 | 외부 데이터 + copy-paste | – | – | 0.45127 / 0.45759 `[확인 필요]` |

Faster R-CNN 최종 점수는 README(#131)에 0.45127, 실험 결과 보고서에 0.45759로 서로 다르게 적혀 있습니다. 어느 실험의 점수인지 확인이 필요합니다.

## 데이터·EDA (main)

| 내용 | 위치 | PR |
|---|---|---|
| 대회 데이터 EDA, 라벨 보정, 분할 | `notebooks/01_eda.ipynb`, `data_process/` | #38, #56 |
| AI Hub 외부 데이터 선별·2차 검수 | `notebooks/02_eda.ipynb`, `data/processed/02_eda_*.csv` | #93, #104, #140 |
| 팀원별 초기 EDA | `notebooks/archive/` | – |

데이터 파이프라인 사용법은 [DATA_PIPELINE.md](DATA_PIPELINE.md)에 있습니다.

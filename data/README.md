# 데이터

## 폴더 구조

```text
data/
├── raw/                        # 원본 데이터
│   ├── train_images/
│   ├── train_annotations/
│   └── test_images/
├── processed/                  # 원본과 분리한 가공 결과 (데이터 파이프라인이 만듦)
│   ├── added_annotations.csv   # 라벨 누락 추가 기록 (kwon)
│   ├── fixed_annotations.csv   # 위치 수정 기록 (kwon)
│   ├── corrections.csv         # 통합 보정 기록 (수정 전후 좌표와 근거)
│   ├── padding_review.csv      # 여백 검수표
│   ├── labels/                 # 라벨 버전 (raw, v1, v2, v2auto)
│   ├── splits/                 # train/val 분할 (fold별)
│   ├── yolo/                   # YOLO 학습 입력 (images, labels, data.yaml)
│   └── ...                     # padding/, results/, predictions/
├── external/                   # 외부 데이터를 푼 폴더 (아래 "외부 데이터")
│   ├── aihub_zips/             # 팀 Drive에서 받은 AI Hub zip
│   └── aihub_combo/<zip 이름>/ # 그 zip을 푼 AI Hub 경구약제 조합 데이터
└── README.md
```

`raw/`, `processed/`, `external/`은 Git에서 제외합니다. 단, `processed/` 바로 아래의 csv(보정 기록,
검수표)는 팀 공유를 위해 커밋할 수 있습니다. 원본은 `raw/`에서 읽기만 하고, 원본 JSON을
고치지 않고 보정 기록으로 라벨 버전을 만듭니다. 만드는 방법과 각 파일의 뜻은
[docs/DATA_PIPELINE.md](../docs/DATA_PIPELINE.md)를 참고하세요.

## 외부 데이터 (AI Hub 경구약제 조합)

- 출처: [AI Hub 경구약제 이미지 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=576)(dataSetSn=576) 중 조합 데이터.
  `TL_2`(라벨)와 그 원천 `TS_2`(이미지)는 쓰지 않습니다. 검증용(`VL_1`·`VS_1`)은 `VS_1`이 팀 Drive에
  올라오기 전까지 `02_eda.ipynb`의 `EXCLUDE`로 뺍니다.
- zip은 팀 Google Drive의 `code_it/pillbbox/aihub_pill/` 폴더에 받은 그대로 둡니다. AI Hub 약관상 팀원에게만 공유합니다.
  AI Hub는 해외 IP 다운로드를 막으므로 Colab에서는 받을 수 없고, 국내 PC에서 AI Hub API(`.env`의
  `AI_HUB_API_KEY`)로 받아 Google Drive 데스크톱의 이 폴더에 넣습니다. 팀에서 한 번만 받습니다.
- `notebooks/02_eda.ipynb`가 이 폴더의 공유 링크(`.env`의 `AIHUB_URL`)에서 zip을 `external/aihub_zips/`로 받고
  `external/aihub_combo/<zip 이름>/`에 풉니다(`data_process/aihub_data.py`). Colab에서는 실행만 하면 됩니다.
  로컬에서 `.env`에 `AIHUB_ZIP_DIR`(예: Drive 데스크톱 폴더)을 적으면 받지 않고 그 폴더를 씁니다.
- JSON 구조는 아래 스키마와 같습니다. 대회 데이터와 같은 로더(`load_annotations`)로 읽습니다.
  단, AI Hub 원본은 `category_id`가 모두 1(`Drug`)이라 `load_combo_annotations`가 대회 데이터처럼
  `drug_N`의 숫자(K-001900 → 1900)로 바꿉니다. `dl_idx`는 약 코드보다 1 작아 클래스로 쓰지 않습니다.
- 지금은 EDA용입니다. 학습 데이터(`data_main`)에는 아직 넣지 않습니다.

## Annotations (JSON 파일 구조)

필수여부: M = 필수, O = 선택, C = 조건부

| 항목명 | 타입 | 필수여부 | 설명 |
|---|---|---|---|
| `images` | Object | M | 약제 이미지정보 |
| `images[].id` | Number | M | 약제 이미지식별자 |
| `images[].width` | Number | M | 약제 이미지너비 |
| `images[].height` | Number | M | 약제 이미지높이 |
| `images[].file_name` | String | M | 약제 이미지파일명 |
| `images[].drug_N` | String | M | 알약코드 |
| `images[].drug_S` | String | M | 알약상태 |
| `images[].back_color` | String | M | 촬영배경 |
| `images[].drug_dir` | String | M | 알약방향 |
| `images[].light_color` | String | M | 촬영조명 |
| `images[].camera_la` | Number | M | 카메라위도 |
| `images[].camera_lo` | Number | M | 카메라경도 |
| `images[].size` | Number | M | 이미지 사이즈 |
| `images[].dl_idx` | String | M | 알약 식별자 |
| `images[].dl_mapping_code` | String | M | 제품코드 |
| `images[].dl_name` | String | M | 제품명 |
| `images[].dl_name_en` | String | O | 제품명(영어) |
| `images[].img_key` | String | M | 이미지 링크 |
| `images[].dl_material` | String | M | 성분명 |
| `images[].dl_material_en` | String | O | 성분명(영어) |
| `images[].dl_custom_shape` | String | M | 제조 모양 |
| `images[].dl_company` | String | M | 제조사명 |
| `images[].dl_company_en` | String | O | 제조사명(영어) |
| `images[].di_company_mf` | String | M | 위탁제조사명 |
| `images[].di_company_mf_en` | String | O | 위탁제조사명(영어) |
| `images[].item_seq` | Number | M | 품목기준코드 |
| `images[].di_item_permit_date` | Date | O | 허가일자 |
| `images[].di_class_no` | String | M | 약품 분류 |
| `images[].di_etc_otc_code` | String | M | 전문의약품/일반의약품 |
| `images[].di_edi_code` | String | M | EDI 코드 |
| `images[].chart` | String | M | 알약 제형 |
| `images[].drug_shape` | String | M | 알약 모양 |
| `images[].thick` | Number | M | 알약 두께 |
| `images[].leng_long` | Number | M | 알약 장축 |
| `images[].leng_short` | Number | M | 알약 단축 |
| `images[].print_front` | String | C | 식별문자_앞 |
| `images[].print_back` | String | C | 식별문자_뒤 |
| `images[].color_class1` | String | M | 색상 1 |
| `images[].color_class2` | String | O | 색상 2 |
| `images[].line_front` | String | M | 알약 앞면분할선 여부 |
| `images[].line_back` | String | M | 알약 뒷면분할선 여부 |
| `images[].img_regist_ts` | Date | O | 약학정보원 이미지 생성일 |
| `images[].form_code_name` | String | M | 정제 분류명 |
| `images[].mark_code_front_anal` | String | M | 알약 앞면마크 형태 |
| `images[].mark_code_back_anal` | String | M | 알약 뒷면마크 형태 |
| `images[].mark_code_front_img` | String | M | 알약 앞면마크 이미지 |
| `images[].mark_code_back_img` | String | M | 알약 뒷면마크 이미지 |
| `images[].mark_code_front` | String | M | 알약 앞면마크 코드 |
| `images[].mark_code_back` | String | M | 알약 뒷면마크 코드 |
| `images[].change_date` | Date | O | 변경일자 |
| `type` | type | O | json 타입 |
| `annotations` | Object | M | 라벨링정보 |
| `annotations[].area` | Number | M | 바운딩박스 면적 |
| `annotations[].iscrowd` | Number | O | 평가 분류 |
| `annotations[].bbox` | List | M | bbox 좌표 |
| `annotations[].category_id` | Number | M | category 아이디 |
| `annotations[].ignore` | Number | O | 무시 여부 |
| `annotations[].segmentation` | List | O | 라벨링바운딩 박스 |
| `annotations[].image_id` | Number | M | 이미지 아이디 |
| `annotations[].id` | Number | M | 어노테이션 아이디 |
| `categories` | Object | M | 라이선스 |
| `categories[].supercategory` | String | M | 슈퍼 카테고리 |
| `categories[].id` | Number | M | 카테고리 아이디 |
| `categories[].name` | String | C | 카테고리 명 |

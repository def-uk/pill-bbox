# 데이터

## bbox 검사·수정 결과 보관

이번 bbox 작업 결과는 프로젝트 루트의 `reports/`에 보관한다.

- `reports/originals/`: 수정 전 JSON 4개 백업(`.json.bak`)
- `reports/corrected_annotations/`: 수정한 JSON 4개의 사본. 전체 학습 라벨이 아니다.
- `reports/bbox_repair/`: 변경 기록, 수정 이미지, 수정 전 검사 결과
- `reports/bbox_audit/`: 수정 후 전체 검사 결과
- `reports/audit_bboxes.ps1`: 현재 `data/raw/`의 학습 이미지와 JSON 검사
- `reports/repair_bboxes.ps1`: 이미 적용한 수정 작업 기록. 재실행용이 아니다.

실제 데이터는 `raw/train_images/`, `raw/train_annotations/`, `raw/test_images/`에 있다.
노트북 경로도 현재 데이터 위치에 맞춰 사용한다.

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

# 학습 및 결과

`train_full.ipynb`는 실행된 전체 학습 셀을 추출한 새 학습용 노트북입니다. Colab GPU에서 설치·Drive 연결, 데이터 복원, 학습 순서로 실행합니다. `resume_training.ipynb`는 중단된 학습 전용이며 두 노트북을 함께 실행하지 않습니다. 20260914_060759 실행은 이미 완료됐으므로 다시 재개하지 않습니다.

데이터: [Roboflow Indoor elder objection v3](https://universe.roboflow.com/zeka-s-workspace/indoor-elder-objection/dataset/3), Zeka's Workspace, CC BY 4.0. YOLOv8 ZIP을 본인 Drive의 `disaster_drone_backup/datasets/indoor_elder_v3_yolov8.zip`에 준비합니다. 원본 16종에서 person/fire/smoke/door/stairs를 5종으로 변환하고 stairs는 staircase로 표시합니다. 픽셀 단위 중복은 test → valid → train 순서로 제거합니다. 원본 데이터·API 키·가중치는 저장소에 포함하지 않습니다.

실행 환경: Tesla T4, Ultralytics 8.4.150, torch 2.11.0+cu128, YOLOv8n, 640×640, batch 16, 30 epochs, seed 42. 21회차 이후 체크포인트의 optimizer를 복원해 22~30회차를 완료했습니다. 매 회차 best.pt/last.pt/results.csv는 SHA-256으로 검증한 뒤 Drive에 저장합니다.

## 최종 독립 테스트

학습 26,031장 · 검증 5,581장 · 테스트 5,581장. 아래 값은 완료된 Colab 로그를 소수점 3자리로 옮긴 값입니다.

| 지표 | 값 |
|---|---:|
| mAP50 | 0.630 |
| mAP50–95 | 0.413 |
| Precision | 0.704 |
| Recall | 0.582 |

| 클래스 | AP50 |
|---|---:|
| person | 0.551 |
| fire | 0.804 |
| smoke | 0.671 |
| door | 0.815 |
| staircase | 0.308 |

원본 결과는 개인 Drive `disaster_drone_backup/full_training/20260914_060759/`의 metrics.json, results.csv, best.pt, summary_chart.png, predictions.jpg에 있습니다. 이는 공개 RGB 데이터 탐지 평가이며 깊이 지도나 Jetson 실측 평가가 아닙니다. 계단 테스트 객체 31개로 표본이 적습니다. 계단·사람 미탐 개선, 유사 장면 누수 점검, 실제 센서 통합 검증이 필요합니다. 다른 분할·이전 모델 결과와 동등 조건의 비교가 아닙니다.

## 맵핑에 모델 연결

신뢰하는 best.pt를 Jetson 로컬 폴더로 복사하고 `mapping/config.json`의 weights를 해당 파일 경로로 설정합니다. CUDA 설치를 확인한 뒤 device를 설정합니다. TensorRT engine은 실제 Jetson 환경에서 변환해야 합니다. 카메라 장착 측정 및 환경 점검 절차는 mapping/README.md를 따릅니다.

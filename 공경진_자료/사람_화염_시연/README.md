# 사람·화염 시연 (2026-09-16)

## 결과물
- `사람탐지_2D맵핑_연동시연.mp4`: 80초. Jetson에서 TUM fr3/walking_xyz RGB-D를 0.1배 입력 속도로 재생하고, 사용자 학습 YOLO와 RTAB-Map을 동시에 실행했다. 실제 관제 UI 녹화 15~175초를 화면 2배속으로 편집했다. 모델 추론·위치추정 속도가 실시간 센서 속도라는 의미는 아니다.
- `화염_학습모델_탐지시연.mp4`: 20초. FURG Fire Car1 원본 0~19.75초를 4fps로 샘플링해 같은 모델로 Jetson CPU에서 추론한 결과를 관제 UI로 시각화했다. RGB 전용이므로 깊이·지도·대상 위치를 생성하지 않았다. 장면에 사람 탐지가 없으면 사람 수는 0으로 표시한다.

두 영상은 서로 다른 입력이다. 사람과 화염이 동일한 RGB-D 장면에서 동시에 탐지·맵핑됐다는 증거가 아니다. 화염 거리·지도 위치 연동은 아직 미검증이다. 현장 D435i·자율 비행 영상이 아니다.

## 실행 및 검증
- 모델 SHA256: bd59f4702d23bcbe03f9381428a4218041f64e8da1b11d10dd5387bedb11982d
- 30회 학습 YOLOv8n을 재학습 없이 사용. confidence 0.30, imgsz 640, CPU 추론. person/fire만 표시.
- 사람 기록 전체: 탐지 메시지 240개, 사람 포함 160개, 지도 좌표 포함 110개. 지도 스냅샷 44개이며 occupancy 배열 해시가 모두 다르다. 이 숫자는 영상에서 선택한 구간만의 수치가 아닌 원본 전체 기록의 집계다.
- 화염 Car1: 80프레임 중 69프레임에서 fire 탐지. 이는 시연 프레임 탐지 여부 집계이며 정확도, AP, recall이 아니다. 오탐·미탐 평가용 수작업 정답 검증은 하지 않았다.
- 원래 barbecue 장면에서는 80프레임 중 5프레임만 탐지되어, 불꽃이 더 명확한 Car1을 시연으로 선택했다. `fire_probe.json`에 추가 장면 표본 결과를 남겼다. 미탐 프레임은 영상에서도 그대로 남긴다.
- 위치추정/수신 대기 상태가 반복되며, 지도 갱신과 사람 좌표는 유효한 깊이와 해당 시각 추정 변환이 있을 때만 표시한다. 사람 좌표는 순간 ROI 추정이며 지속적인 사람 추적 또는 영구 시맨틱 지도는 아니다.
- 기존 처리 테스트 14개 통과, 두 MP4 전체 FFmpeg 디코딩 검사 통과.
- 완료 후 Jetson 실시간 D435i 관제 화면을 다시 실행했다.

## 자료 출처
- TUM RGB-D benchmark, fr3/walking_xyz: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- Freiburg3 RGB intrinsics 및 5000 depth units/m: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
- FURG Fire dataset, Car1.mp4, CC0-1.0 repository: https://github.com/steffensbola/furg-fire-dataset
- Steffens, Botelho, Rodrigues, An Unconstrained Dataset for Non-Stationary Video Based Fire Detection (2015).

원본 녹화 `people_console.mp4`, 지도/위치/탐지 로그 `people_evidence/`, 원본 화염 프레임과 예측 `fire_inference/`, 집계 `verification.json`에 보존했다.

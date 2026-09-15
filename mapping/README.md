# 한성공학경진대회 실내 재난 인식·RGB-D 지도 시스템

기존 코드의 깊이 처리와 RTAB-Map 실행 구성을 재사용해 실제 입력 처리, 네이티브 데스크톱 화면, 저장 및 종료를 연결했습니다. **소프트웨어 구현과 Windows 기록 데이터 검증을 완료한 상태이며, Jetson+D435i 실물 통합 검증은 아직 수행하지 않았습니다.** 장비 없이 SLAM이나 5종 탐지가 검증되었다고 주장하지 않습니다.

F450은 전시 장식입니다. 비행 제어, Pixhawk, MAVLink, Optical Flow 명령은 없습니다. Colab·Drive 학습 작업, 포스터·Canva 및 GitHub는 수정하지 않았습니다.

## 지금 Windows에서 실행

현재 작업 컴퓨터에는 전용 가상환경을 `../../work/disaster_venv`에 준비했습니다. **`화면실행.cmd`를 더블클릭**하면 입력이 없는 대기 화면이 열립니다. 합성 영상·탐지·지도는 만들지 않습니다.

공개 실제 RGB-D 기록을 화면에서 보려면 이 폴더의 PowerShell에서 실행합니다.

```powershell
.\start_windows.ps1 -Source tum -Dataset '..\..\work\public_rgbd\rgbd_dataset_freiburg1_xyz'
```

이 기록은 Microsoft Kinect로 실제 사무실을 촬영한 TUM fr1/xyz RGB-D 시퀀스입니다. 영상, Depth, 카메라 기준 격자만 표시합니다. **RTAB-Map 누적 지도 영역은 비워 둡니다.** 입력 종료 후 마지막 프레임에 STALE이 표시됩니다. 영상은 원본 촬영 시각에 맞춰 재생하고, 깊이 처리는 최신 프레임을 별도 주기로 소비합니다. 기본 1Hz로 저장하여 녹화를 위한 파일 쓰기가 전체 재생 속도를 늦추지 않도록 했습니다. 모델을 연결하면 탐지 영상은 설정한 추론 주기를 따릅니다.

다른 Windows 컴퓨터에서는 Python 3.12 설치 후 `setup_windows.ps1`을 실행하면 프로젝트 전용 `.venv`가 만들어집니다. PowerShell 실행 정책으로 차단되면 다음 명령을 사용합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

Windows에서 나중에 D435i를 연결해 깊이만 먼저 확인하려면 해당 가상환경에 Python 버전과 맞는 `pyrealsense2`를 설치하고 `start_windows.ps1 -Source sdk`를 실행합니다. SDK 모드는 공식 align-depth2color 예제의 정렬 처리에 기반하며 **카메라 기준 투영만 수행합니다.** ROS 모드와 동시 실행하지 않습니다.

## Jetson에서 처음 한 번 준비

1. 이 `disaster_system` 폴더를 Jetson에 복사합니다. Windows `.venv`는 복사하지 않습니다.
2. 아래 환경 정보를 확인합니다. RAM과 JetPack을 추정해 CUDA·torch를 재설치하는 스크립트는 없습니다.

```bash
cat /etc/os-release
cat /etc/nv_tegra_release
free -h
python3 --version
```

3. 실제 Ubuntu/JetPack에 맞는 ROS 2, RealSense SDK/ROS 드라이버, RTAB-Map을 준비합니다. **Ubuntu 22.04와 ROS 2 Humble가 확인되고 ROS apt 저장소가 이미 설정된 경우에만** 아래 예시가 해당됩니다.

```bash
sudo apt install ros-humble-rtabmap-ros ros-humble-imu-filter-madgwick \
  ros-humble-cv-bridge ros-humble-message-filters ros-humble-tf2-ros \
  ros-humble-rosbag2 python3-tk python3-pil python3-numpy python3-opencv
```

RealSense 드라이버/커널 지원은 공식 [ROS wrapper](https://github.com/realsenseai/realsense-ros) 및 [Jetson 설치 지침](https://dev.realsenseai.com/docs/nvidia-jetson-tx2-installation)의 실제 사용 JetPack 지원 범위를 확인합니다. 오래된 TX2 지침의 명령을 Orin NX에 무조건 적용하지 않습니다. 기존 동작 환경이 있으면 유지합니다.

4. `rclpy`, `cv_bridge`, `torch`, `torchvision`, `ultralytics`를 함께 import할 수 있는 Python을 선택합니다. Jetson용 torch는 해당 JetPack에 맞는 NVIDIA 배포 환경을 사용합니다. Windows용 requirements를 Jetson에 일괄 적용하지 않습니다. GPU 확인 전 기본 `device`는 `cpu`입니다. 설치된 GPU torch의 CUDA 동작을 확인한 후 `"device": "0"`으로 바꿉니다.
5. 아래 좌표계 설명에 따라 실제 카메라 장착 값을 측정해서 `config.json`에 넣고 `mount_verified`를 `true`로 변경합니다. **기본 높이 1m는 예시이며 실측값이 아닙니다.** 미확인 상태에서는 실제 ROS 실행을 환경 점검이 중단합니다. 이는 잘못된 지면 기준 지도를 방지하는 설정 검사입니다.

## 한 번 실행

설치된 ROS 환경을 source한 터미널에서 다음 한 명령이 카메라, IMU 필터, 위치추정, SLAM 입력 게이트, YOLO, 녹화, 통합 화면을 시작합니다.

```bash
source /opt/ros/humble/setup.bash  # 실제 설치 배포판에 맞게 변경
bash start_jetson.sh
```

별도의 Python 환경을 사용한다면 `DISASTER_PYTHON=/실제/환경/bin/python bash start_jetson.sh`로 지정합니다. 모든 사용자 Python 처리 노드가 같은 인터프리터를 사용합니다. ROS CLI 자체도 설치된 시스템 환경에서 작동해야 합니다.

사전 점검만 수행: `bash start_jetson.sh --check`. 설치나 카메라 스트림 시작은 하지 않습니다. 화면 모드는 Linux 데스크톱/X11이 필요하며 headless 기록 모드는 디스플레이가 필요 없습니다. 기본 ROS_DOMAIN_ID는 73이며, 이미 환경 변수에 설정된 값은 유지합니다. 외부에서 토픽을 확인할 때 같은 값을 사용합니다.

### 회수 후 확인하는 전시 운영

**실제 운영은 Jetson 단독입니다. 현장에서 계산·맵핑·저장하고 회수 후 Jetson에 별도 모니터를 연결해 결과를 봅니다. PC 연결, 원격 기능, 자동복귀는 필요하지도 포함되지도 않습니다.** 이미 Jetson에 준비된 키보드·마우스로 조작합니다. 외부 모니터 없이 기록하려면:

```bash
bash start_jetson.sh --headless
```

이 상태에서 종료는 같은 Jetson의 다른 터미널에서 `python3 stop.py` 또는 원래 터미널의 Ctrl+C입니다. 저장 마감이 끝난 뒤 전원을 끕니다. 단순 전원 차단은 bag/DB 마감을 보장하지 못합니다. 자율비행·자동복귀 명령은 보내지 않습니다.

회수 후 **Jetson에 모니터를 연결한 뒤** 다음 한 명령을 실행합니다. 진행 중인 기록에 정상 종료를 요청하고 저장 마감 후 최근 결과를 엽니다.

```bash
bash view_results.sh
```

과거 날짜 선택: `python3 run.py --source session --session runs/날짜_시각`. PC로 파일을 옮길 필요는 없습니다. Windows의 ‘지난기록확인.cmd’는 개발용 부가 기능이며 실제 전시 운영 경로가 아닙니다.

기록 확인 화면에는 저장된 마지막 지도, 보정된 키프레임 궤적, 마지막 저장 RGB/Depth와 같은 시각의 탐지를 표시합니다. 전체 영상은 `frames/` 이미지 시퀀스와 `rosbag/`에 남습니다. 기록 확인 모드는 지도 재계산이 아니라 저장 결과 열람이며 **RECORDED**로 표시합니다. 원본 세션을 수정하지 않고, 열람 화면을 따로 저장하면 새로운 세션에 보관합니다. 기본 실행은 자동 부팅 서비스가 아니므로 전원 켠 뒤 명령을 한 번 실행해야 합니다.

한 번 찍어 둔 공학관 기록으로 알고리즘을 다시 시험하려면 `bash start_jetson.sh --bag runs/날짜_시각/rosbag`을 실행합니다. 저장된 센서 토픽과 IMU만 재생하고 map/odom/탐지 결과를 답으로 재생하지 않습니다. ROS clock을 사용하고 기록의 내부 카메라 TF만 전달합니다. 이 ROS bag 재처리 경로는 코드·문법 확인 단계이며 Jetson 실물에서 검증해야 합니다.

### 가상환경에서 맵핑 시험 및 MP4

`simulation/virtual_mapping.mp4`는 가상 RGB-D 180프레임을 OpenCV RGB-D odometry에 넣어 위치를 추정하고 점유 격자를 누적한 실행 결과입니다. **SIMULATION ONLY**로 표시하며 실제 D435i·RTAB-Map·YOLO 검증 영상으로 사용하지 않습니다. 물체와 방은 프로그램이 생성한 가상 공간이며 한성대 공학관 실제 구조가 아닙니다.

다시 만들려면 Windows 프로젝트 가상환경에서:

```powershell
..\..\work\disaster_venv\Scripts\python.exe virtual_test.py --out simulation_new --frames 180 --fps 15
```

생성된 RGB/Depth는 `simulation/virtual_rgbd/`에 함께 저장됩니다. Jetson에서 **생산용 RTAB-Map 파이프라인 자체**에 가상 입력을 넣으려면:

```bash
bash start_jetson.sh --config simulation/virtual_config.json --replay simulation/virtual_rgbd
```

OpenCV 시험은 루프 폐쇄 없는 위치추정 기반 점유 셀 누적입니다. 프로그램 내부 정답 경로는 오차 비교에만 사용하고 추정 지도에는 사용하지 않았습니다. 자세한 결과는 검증결과.md에 있습니다.

**‘저장하고 종료’ 또는 Esc**를 누르면 프로그램이 저장하고, 카메라 및 처리 프로세스에 SIGINT를 보내 RTAB-Map DB와 rosbag을 마감합니다. 터미널 Ctrl+C도 지원합니다. 강제 종료가 필요했던 경우 `shutdown.json`에 표시되며 DB/bag 무결성을 별도로 확인해야 합니다.

## 모델 교체 — 학습과 완전히 분리

프로그램은 Drive에 접근하거나 모델을 다운로드하지 않습니다. 사용자가 신뢰하는 학습 모델 파일을 Jetson의 로컬 폴더로 복사하고 `config.json`의 `weights`만 바꿉니다.

```json
"weights": "/home/사용자/models/best.pt"
```

- 현재 시험 모델 원본: `disaster_drone_backup/visual_trial/best.pt`
- 최종 모델 원본: `disaster_drone_backup/full_training/<실행 날짜 폴더>/best.pt`
- 클래스 순서는 고정하지 않지만, 모델 이름이 정확히 person/fire/smoke/door/staircase 5종인지 검사합니다.
- 모델이 없거나 잘못되면 NO_MODEL/MODEL_ERROR 상태를 표시합니다. RGB-D와 지도 처리는 계속할 수 있습니다. 탐지 결과를 대신 만들어 내지 않습니다.
- 교체 후 프로그램을 다시 실행합니다. 세션 기록에 모델 경로와 SHA-256을 남겨 어느 가중치의 결과인지 구분합니다.

## 데이터 흐름 및 시간 동기화

```text
D435i ─ realsense2_camera (USB를 여는 유일한 프로세스)
            ├─ RGB + aligned Depth + CameraInfo ─ YOLO ─ 영상/거리/JSON
            ├─ RGB-D odometry + IMU ─ odom / odom_info
            └─ RGB-D + 같은 시각의 odom / odom_info
                  └─ slam_gate ─ RTAB-Map ─ /map, /disaster/map_graph, TF
모든 결과 ─ 네이티브 Tk 화면 + 세션 파일 + rosbag
```

ROS 드라이버의 `align_depth.enable=true`, `enable_sync=true`를 사용합니다. RGB·정렬 Depth·CameraInfo가 같은 optical frame, 같은 영상 크기인지 확인합니다. 입력 허용 시간차는 기본 15ms이고 30Hz 입력에서 인접 프레임과 섞이지 않도록 설정했습니다. 큐는 유한하며 처리량이 부족하면 프레임을 생략합니다. 지연된 모든 프레임을 끝없이 쌓지 않습니다. 드라이버가 다른 토픽/프레임 이름을 발행하면 조용히 사용하지 않고 대기/오류가 나타나므로 버전별 토픽을 확인해야 합니다.

거리 계산에서 ROS `16UC1`은 mm→m, `32FC1`은 m로 처리합니다. SDK 직접 입력은 장치의 실제 depth scale을 읽습니다. 모델의 박스 중앙 절반 영역에서 유효 깊이의 중앙값을 계산합니다. 이는 광축 방향 Z 추정값이며 사람이 줄자로 잰 최단거리 또는 박스 전체의 거리라는 뜻은 아닙니다. 깊이가 부족하면 미확정입니다. 화염·연기는 `depth_m=null`로 유지하고, 뒤쪽 표면일 수 있는 ROI 값은 `roi_surface_depth_m`에 별도로 기록합니다.

## 좌표계와 지도 의미

- `map → odom`: RTAB-Map의 루프 폐쇄 보정.
- `odom → base_link`: RGB-D odometry. 위치추정 자동 리셋을 비활성화해 실패 후 원점으로 갑자기 바뀐 위치를 정상 궤적으로 잇지 않습니다.
- `base_link → camera_link`: `mount_xyz_m` / `mount_rpy_rad`. base는 카메라를 지지하는 장치의 기준점이며 초기 지면 높이를 z=0으로 둡니다. 기본 예시 카메라는 그 점에서 위로 1m이고 앞을 향합니다. 이동 중 고정된 장착 변환이어야 합니다. 손으로 카메라만 들면 초기 카메라 아래 기준점을 정의하고 그 가상 기준 프레임도 카메라와 함께 움직이는 것으로 해석합니다.
- `camera_link → camera_color_optical_frame`: RealSense 드라이버의 실측 내부 extrinsic TF를 사용합니다. 중복 TF를 발행하지 않습니다.
- body 좌표 x=앞, y=왼쪽, z=위. optical 좌표 x=오른쪽, y=아래, z=앞. roll/pitch/yaw는 라디안, 장착 translation은 미터입니다.

누적 지도는 직접 격자를 과거 프레임 위에 겹치는 방식이 아니라 **RTAB-Map OccupancyGrid**입니다. 5cm 해상도, 최대 6m, 지면·장애물 분리와 ray tracing을 사용합니다. 기본 지면 상한 0.15m, 장애물 상한 2m는 초기 지면 기준입니다. 장착 보정과 IMU 초기 자세가 틀리면 이 분리도 틀려질 수 있습니다. 지도는 미관측/관측된 빈 공간/점유를 구분하며 안전한 비행 공간 또는 계단 이동 가능 지도가 아닙니다.

별도 작은 패널의 카메라 기준 투영은 기존 `mapping.py`를 재사용한 것입니다. `local_height_band_m`로 카메라 중심 대비 위쪽 양의 높이 범위를 선택합니다. 기본 -0.35~+0.35m의 점만 표시하며 **누적하지 않습니다**. 빈 셀은 미관측입니다.

지도 궤적은 RTAB-Map graph의 최적화된 키프레임 위치로 매번 다시 그려 루프 폐쇄를 반영합니다. 모든 프레임의 연속 궤적은 아니며 현재 카메라 위치는 해당 시각 TF로 별도 표시합니다. 탐지 지도 마커도 영상 시각의 TF와 보정된 카메라 광선을 사용합니다. TF가 없거나 위치추정이 실패하면 표시하지 않습니다. 마커는 현재 탐지의 ROI 중심 근사이며, 영구적인 객체 위치 지도는 아닙니다. 과거 객체를 보정 없이 누적하지 않습니다.

## 실패·기록 동작

`slam_gate`는 RGB·Depth·CameraInfo·Odometry·OdomInfo를 시간으로 묶고 lost/covariance/quaternion/타임스탬프를 검사한 성공 프레임만 SLAM에 전달합니다. 실패 또는 입력 중단 시 새로운 입력을 전달하지 않습니다. 이미 처리 중이던 마지막 정상 프레임은 RTAB-Map에서 마무리될 수 있습니다. 화면은 LOST/STALE을 표시하고 현재 위치·마커를 숨깁니다. 마지막 실제 지도는 유지하되 정상 실시간 결과로 표시하지 않습니다. 타임스탬프가 역행하면 새 세션을 시작해야 합니다.

모든 실행은 `runs/<날짜_시각>/` 새 폴더를 만들며 이전 지도를 삭제하지 않습니다.

| 파일 | 내용 |
|---|---|
| `rtabmap.db` | RTAB-Map 누적 지도 DB, 정상 종료 때 마감 |
| `map.npz` | 원본 점유 값과 같은 파일 안의 해상도·원점·시각 |
| `map.pgm` / `map.yaml` / `map.png` | 재사용 가능한 2D 지도 및 보기용 이미지 |
| `optimized_path.json` | 루프 폐쇄가 반영된 카메라 키프레임 궤적 |
| `poses.jsonl` | 시각별 TF 카메라 위치; 기록 당시 map 보정 기준 |
| `detections.jsonl` | 클래스, 신뢰도, 박스, 거리 상태, 시각, 현재 지도 마커 |
| `frames/` | 기본 1Hz RGB 이미지와 실제 Depth/K/D/시각 NPZ |
| `rosbag/` | 기본 활성화. RGB·Depth·CameraInfo·IMU·TF·지도·탐지 영상/결과 전체 토픽 기록 |
| `snapshot_*/` | 저장 버튼/종료 시 마지막 실제 패널 이미지와 상태 |
| `environment.json`, `manifest.json` | OS/RAM/ROS/Python/CUDA 확인 결과, 설정, 모델 해시 |
| `tracking.jsonl`, `pipeline.log`, `recording.log`, `shutdown.json` | 상태 변화, 프로세스 오류, 종료 결과 |

영상은 시간 정보를 보존하는 rosbag 및 이미지 시퀀스로 저장합니다. 고정 FPS MP4로 처리 지연을 감추지 않습니다. RGB-D 전체 bag은 저장량이 크므로 전시 전 여유 공간을 확인하고, 불필요하면 `record_bag=false`로 설정합니다. 부분 이미지/Depth 기록은 계속됩니다.

## 공개 데이터와 실물 검증

[검증결과.md](검증결과.md)에 실제 수행 범위를 기록했습니다. 공개 데이터 재검증:

```bash
python validate_public_rgbd.py --cache /원하는/캐시 --out validation/public_new
python -m unittest discover -s tests -v
```

ROS가 설치된 Linux에서는 추출된 공개 샘플을 원래 ROS 입력 토픽으로 재생할 수 있습니다.

```bash
bash start_jetson.sh --replay /캐시/rgbd_dataset_freiburg1_xyz
```

이 모드에서는 D435i 드라이버와 IMU 필터를 열지 않습니다. RGB·Depth 실측 시각을 일대일로 연결하고 최대 15ms의 시간차를 허용합니다. 원본 시각 간격을 보존하면서 현재 ROS 시계로 공통 오프셋만 적용해 재생합니다. PNG 깊이는 5000으로 나누며, TUM 공식 권고에 따라 정렬된 영상의 기본 intrinsics(525/525/319.5/239.5, distortion=0)를 사용합니다. ground-truth/trajectory 파일을 위치추정 결과로 사용하지 않습니다. **Windows에서는 ROS 재생과 SLAM 자체를 실행 검증하지 않았습니다.** fr1/xyz는 주로 이동 디버깅용이므로 루프 폐쇄 평가는 별도의 방 순환 기록과 실물 시험이 필요합니다.

## Jetson 실물 합격 확인

1. `--check` 통과 및 실제 모델 이름·CUDA 확인. 카메라 RGB 640×480, 정렬 Depth, CameraInfo, IMU 수신 확인.
2. 카메라를 천천히 움직여 TRACKING, 현재 카메라 위치, 키프레임 궤적, `/map` 생성 확인. 지도는 최초 수 초 이상 늦을 수 있습니다.
3. 줄자로 확인한 1m/2m/3m의 불투명 평면을 비교해 깊이 오차를 기록. 화염·연기는 거리 정확도 측정 대상으로 삼지 않습니다.
4. 실제 벽과 장애물 배치를 비교하고, 한 바퀴 돌고 돌아와 루프 폐쇄 후 지도·궤적 정합 확인. 측정 결과를 남기기 전에는 정확도 수치를 주장하지 않습니다.
5. 카메라를 가리거나 USB를 분리하여 LOST/STALE, 지도 입력 차단, 위치·마커 숨김 확인. 다시 보이게 해 복구를 확인하며, 복구하지 못하면 새 세션으로 재시작.
6. 종료 후 `rtabmap.db`, `map.npz`, `ros2 bag info runs/.../rosbag`, 탐지 로그·이미지 개수·모델 해시 확인. 관측하지 않은 공간을 자유 공간으로 표현하지 않는지 확인.

출처 및 라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md)를 참고하세요.

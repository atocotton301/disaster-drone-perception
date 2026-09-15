# 출처·라이선스·재사용 내역

기존 재난 드론 프로젝트의 깊이 처리 및 ROS 연결 코드를 재사용했습니다.

- `mapping.py`: 기존 깊이 정리/박스 중앙값/투영/로컬 격자 함수를 복사하고 높이 범위 설정을 추가했습니다.
- `app.py`: 기존 입력·추론·거리 구조를 읽고 실제 입력 분리 및 클래스 검사를 재사용했습니다. 합성 입력 분기는 가져오지 않았습니다.
- 기존 `ros2/indoor_rgbd.launch.py`: RTAB-Map 공식 예제 기반 구성과 라이선스를 유지하고, 한 번 실행·입력 게이트·설정/저장/장착 TF를 추가했습니다.
- 기존 `ros2/detection_node.py`: 정렬 RGB-D를 구독하는 탐지 처리를 확장했습니다.
- `desktop_dashboard.py`: 검토했으나 합성 데모 UI는 새 실행 경로에 포함하지 않았습니다.
- 기존 학습 코드·데이터 파일은 수정하거나 복사하지 않았습니다.

| 외부 프로젝트 | 출처 | 라이선스 및 사용 |
|---|---|---|
| RTAB-Map ROS | https://github.com/introlab/rtabmap_ros | BSD-3-Clause, `third_party/RTABMAP_LICENSE.txt` 유지 |
| D435i color 예제 | https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_examples/launch/realsense_d435i_color.launch.py | 기존 확인 커밋 `73c98f87a807dd48d3f7d67ff18d77b4c69ac56c`; 최신 ros2 소스와 메시지 정의도 2026-09-14 확인 |
| RealSense ROS | https://github.com/realsenseai/realsense-ros | Apache-2.0, 외부 드라이버; `third_party/REALSENSE_APACHE_2.0.txt` |
| RealSense Python 정렬 예제 | https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py | Apache-2.0, Copyright(c) 2017 RealSense, Inc. 공식 예제의 입력·scale·align 패턴 사용, `inputs.py`에 수정 사항 표시 |
| Ultralytics | https://github.com/ultralytics/ultralytics | AGPL-3.0 또는 별도 Enterprise 조건. 패키지·가중치는 포함하지 않음. 배포 시 실제 설치 버전과 프로젝트 조건 확인 |
| OpenCV | https://github.com/opencv/opencv | 사용 버전의 Apache-2.0 및 포함 의존성 조건 적용; 외부 Python 패키지 |
| OpenCV RGB-D odometry | https://docs.opencv.org/4.12.0/df/ddc/classcv_1_1rgbd_1_1Odometry.html | 가상환경 시험에서 RgbdICPOdometry 사용. 반환하는 source→destination 변환을 역변환하여 카메라 포즈에 누적 |
| FFmpeg / imageio-ffmpeg | https://github.com/imageio/imageio-ffmpeg | 영상 인코딩용 외부 도구. Python wrapper BSD-2-Clause, 실제 배포 바이너리의 FFmpeg 라이선스 조건은 해당 패키지 참조. 바이너리 자체는 결과 ZIP에 미포함 |
| Pillow | https://github.com/python-pillow/Pillow | HPND 계열, 외부 Python 패키지 |
| NumPy | https://github.com/numpy/numpy | BSD-3-Clause, 외부 Python 패키지 |
| TUM RGB-D benchmark | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | CC BY 4.0. J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, “A Benchmark for the Evaluation of RGB-D SLAM Systems”, IROS 2012. Kinect 실측 fr1/xyz 입력을 읽고 분석·부분 이미지 출력 |
| TUM 데이터 포맷 | https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats | 깊이 PNG 5000 units/m, RGB·Depth preregistration 및 권고 기본 intrinsics 참조 |

공개 기록 다운로드 주소와 SHA-256은 `validation/public/public_validation.json`에 있습니다. 공식 홈페이지의 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)에 따라 출처를 표시하며, 깊이 필터링·투영·이미지 변환을 수행했습니다. 원본의 groundtruth를 추정값으로 사용하지 않습니다. 다운로드 원본은 work 캐시에 있으며, 배포 패키지에는 검증 결과와 인용된 일부 프레임만 포함합니다.

초기 검토한 Open3D Redwood living-room1은 합성 ICL-NUIM 계열임을 확인하여 실측 검증 자료 및 최종 실행 모드에서 제외했습니다. 최종 실측 검증에는 TUM을 사용합니다.

외부 패키지는 프로젝트 전체에 일괄 재라이선스하지 않습니다. 배포 ZIP에는 모델, CUDA, torch, ROS 라이브러리를 포함하지 않습니다.

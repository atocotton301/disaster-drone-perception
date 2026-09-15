# disaster-drone-perception

Jetson Orin NX + Intel RealSense D435i를 위한 실내 재난 5종 탐지·RGB-D 지도 시스템입니다. F450은 전시용 기체이며 자율비행 제어는 포함하지 않습니다.

| 폴더 | 내용 |
|---|---|
| [training](training/README.md) | 실제 Colab 전체 학습·중단 복구 노트북, 최종 테스트 결과와 모델 연결 방법 |
| [mapping](mapping/README.md) | 센서 입력, 탐지·거리, RTAB-Map 연결, 통합 화면, 기록·조회 실행 코드 |

전체 30회 학습 후 공개 테스트 5,581장에서 mAP50 **0.630**, mAP50–95 **0.413**을 기록했습니다. 계단 AP50은 0.308로 보완이 필요합니다. 이는 RGB 탐지 평가이며 실제 Jetson·D435i 통합 성능이나 비행 성능 검증이 아닙니다.

게시 준비 중 Windows 처리 단위 테스트 14개를 재실행해 통과했습니다. ROS/RTAB-Map 및 Jetson 실물 통합은 별도 현장 검증이 필요합니다. `mapping`의 가상 시나리오 코드는 시뮬레이션이며 실제 센서·YOLO 결과로 제시하지 않습니다.

Windows: `mapping/setup_windows.ps1`로 환경 준비 후 `mapping/화면실행.cmd` 실행. Jetson: [설치·장착 확인 절차](mapping/README.md)를 마친 뒤 `bash start_jetson.sh`. 모델 파일을 복사하고 config.json의 weights를 설정해야 탐지가 작동합니다.

모델·원본 데이터·개인 기록·가상환경은 포함하지 않습니다. 문서에 언급된 검증 이미지와 생성 영상은 로컬 산출물이며 저장소에는 원본 코드만 포함합니다. 외부 프로젝트의 출처와 라이선스는 [THIRD_PARTY.md](mapping/THIRD_PARTY.md)를 참고하세요. Ultralytics 사용 시 AGPL-3.0 또는 해당 Enterprise 조건을 확인해야 합니다.

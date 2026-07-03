# AI City Challenge 2026 Track 1 (Multi-Camera 3D Perception / MTMC) — Baseline

RGB-only, YOLO(COCO 사전학습) 검출 + 2D 트래킹 + 지면 호모그래피 기반 3D 역투영 +
좌표 기반 크로스카메라 매칭으로 구성된 **작동하는 베이스라인**. Track1 제출 포맷(`track1.txt`)
까지 end-to-end 로 실행 및 검증 완료.

## 1. 실행 방법

### 환경 설치
```bash
cd /home/kim/aicity2026/track1_3dperception
pip3 install --break-system-packages --user torchvision==0.20.1 ultralytics lap motmetrics
```
(torch==2.5.1+cu121, numpy, opencv-python 은 이미 설치되어 있음. requirements.txt 참조)

### 전체 파이프라인 한 번에 실행
```bash
./run_pipeline.sh --scene Warehouse_000 --split train \
    --cameras "Camera_0000 Camera_0003 Camera_0007 Camera_0011 Camera_0015" \
    --max-frames 250 --device cuda --out track1.txt
```
`--cameras` 생략 시 해당 scene의 모든 카메라, `--max-frames` 생략 시 비디오 전체 프레임 사용.

### 단계별 실행 (디버깅/검증용)
```bash
python3 detect.py       --scene Warehouse_000 --cameras Camera_0000 Camera_0003 --max-frames 250 --device cuda
python3 track2d.py      --scene Warehouse_000 --cameras Camera_0000 Camera_0003
python3 project3d.py    --scene Warehouse_000 --cameras Camera_0000 Camera_0003
python3 fuse_mtmc.py    --scene Warehouse_000
python3 export_submission.py --scene Warehouse_000 --out track1.txt
```
각 단계는 `cache/{detections,tracks2d,tracks3d,global_tracks}/` 아래에 중간 결과를 JSON 캐시로 저장한다.

## 2. 파이프라인 아키텍처

```
video (Camera_XXXX.mp4)
        |
        v
  detect.py  ---- YOLO11n (COCO pretrained), per-frame 2D bbox+conf+coco_class
        |         COCO class -> target class 약한 매핑 (common.COCO_TO_TARGET)
        v
  cache/detections/<scene>__<camera>.json
        |
        v
  track2d.py ---- IoU(lap Hungarian) + Constant-Velocity Kalman filter
        |         camera 별 독립적으로 로컬 track_id 부여
        v
  cache/tracks2d/<scene>__<camera>.json
        |
        v
  project3d.py --- bbox bottom-center 픽셀 -> 역-호모그래피 -> world (X,Y), Z=0
        |          class별 크기 프라이어로 w,l,h 및 center-height z 부여, yaw=0
        v
  cache/tracks3d/<scene>__<camera>.json
        |
        v
  fuse_mtmc.py --- 프레임별 카메라 쌍 간 (x,y) 거리 + 클래스 일치 -> Hungarian 매칭
        |          Union-Find로 (camera, local_track_id) 를 전역 object_id 로 병합
        v
  cache/global_tracks/<scene>.json
        |
        v
  export_submission.py --- 11-컬럼 포맷으로 track1.txt 작성 + 유효성 검사
        v
     track1.txt
```

## 3. calibration.json 행렬 해석 (검증 결과)

`Warehouse_000/calibration.json`, `Camera_0003`, `cameraMatrix`(P, 3x4) / `homography`(H, 3x3):

- **`homography` 는 `cameraMatrix`에서 Z-계수 컬럼(인덱스 2)을 제거한 것과 정확히 일치**한다.
  즉 `H[:, :2] == P[:, [0,1]]` 그리고 `H[:, 2] == P[:, 3]`. 실제 값:
  ```
  P row0 = [ 9.12295, -18.68720, -1.61810, -764.40733 ]
  H row0 = [ 9.12295, -18.68720,           -764.40733 ]
  ```
  → `cameraMatrix` 는 world `(X,Y,Z,1) -> pixel homogeneous (u,v,w)` 의 완전한 3D 투영이고,
  `homography` 는 그 중 `Z=0` 평면(지면)에 대한 투영(3x3, `(X,Y,1)->(u,v,w)`)이다. 둘 다
  **world -> pixel** 방향이며, 픽셀->지면 역변환은 `H`의 역행렬을 사용한다
  (`common.CameraModel.pixel_to_ground`).

- **`ground_truth.json`의 `3d location`은 3D 박스의 중심점**(바닥점이 아님)임을 확인:
  `Camera_0003`, frame 0, PalletTruck(object id 2833), `3d location.z = 0.9787`,
  `scale.h = 1.9651` → `z ≈ h/2` (박스 중심 높이). 박스 바닥 `Z = z - h/2 ≈ -0.0038`(거의 0)을
  `cameraMatrix`로 투영하면 `(1509.9, 704.7)`인데, 이는 `homography`(Z=0 가정)로 투영한 값과
  거의 동일했고, 실제 2D bbox `[1409,456,1577,739]`의 바닥-중앙점 `(1493, 739)`과도 약 20-35px
  이내로 근접함 — 즉 **bbox bottom-center ≈ 지면 접촉점**이라는 가정이 유효함을 확인했다.

- 역방향 검증(`pixel_to_ground` → `ground_to_pixel` 라운드트립)도 픽셀 좌표를 정확히 복원함을
  확인 (`common.py` 스모크 테스트에서 `(1493.0, 739.0) -> world -> (1493.0, 739.0)`).

이 검증 내용을 바탕으로 `common.py::CameraModel` 에 `project_point3d` (full 3D, world->pixel),
`pixel_to_ground` (pixel->world Z=0, `H`의 역행렬), `ground_to_pixel` (world Z=0->pixel) 세
헬퍼를 구현했다.

## 4. 한계점

- **검출기가 COCO 사전학습 YOLO11n** 이라 웜하우스 특화 클래스(Forklift, PalletTruck,
  NovaCarter, Transporter, FourierGR1T2, AgilityDigit)에 대한 직접적인 대응 클래스가 없음.
  `person`만 신뢰할 수 있는 대응(Person)이고, `truck->Forklift`, `car->PalletTruck`,
  `motorcycle->NovaCarter`는 **약한 프록시**(형태만 대략 비슷한 COCO 클래스를 억지로 매핑)라
  정밀도/재현율이 매우 낮음. 실제 실행 결과 확인된 로컬 데이터(Warehouse_000-006)에는
  Person/Forklift/PalletTruck 세 클래스만 존재했고, 스모크 테스트에서도 이 세 클래스만
  검출/제출되었다 (`track1.txt` class_id 0, 1만 등장, 1581줄 중 1514줄이 Person).
- **ReID 없는 순수 거리 기반 크로스카메라 매칭**: 같은 프레임에서 다른 카메라의 검출 간
  `(x,y)` 유클리드 거리(<=2m, 클래스 일치)만으로 매칭. 밀집 구간이나 겹치는 궤적에서 ID
  스위칭/오매칭 발생 가능. `fuse_mtmc.py::pair_cost()`에 ReID embedding 거리를 추가할 수
  있도록 훅 포인트를 마련해 둠 (주석 참조).
- **yaw = 0.0 고정**: 헤딩 추정 로직 없음 (v1 근사).
- **z-height 프라이어 부정확**: 클래스별 평균 높이의 절반을 중심 z로 사용 — 실제 객체별
  포즈/적재 상태에 따른 변동은 반영 못함.
- **클래스별 검출 성능 편차 큼**: Person 검출은 비교적 안정적이나 Forklift/PalletTruck
  프록시는 매칭이 우연에 가까움 (스팟체크에서 우연히 근접한 경우가 있었으나 이는 통계적으로
  신뢰할 수 없음, 아래 5절 참조).
- 2D 트래커는 self-contained 최소 구현(IoU+Kalman, ByteTrack의 2단계 매칭은 부분적으로만
  모사)이라 장기 가림(occlusion)에서 track fragmentation 발생 가능.

## 5. 다음 개선 아이디어

1. **YOLO 파인튜닝**: `ground_truth.json`의 `2d bounding box visible` + `object type`을
   라벨로 사용해 웜하우스 전용 7-클래스 검출기로 파인튜닝 (가장 임팩트 큰 개선).
2. **ReID 임베딩 추가**: crop 이미지 임베딩(예: OSNet/torchreid)을 `fuse_mtmc.py`의 매칭
   cost에 결합 (`alpha*dist + beta*(1-cos_sim)`), NVIDIA MTMC 우승팀들의 표준 접근
   (BoT-SORT + torchreid 조합, 2025 Track1 공개 솔루션 확인).
3. **칼만필터 기반 3D 트래킹**: 현재는 2D 트래킹 후 3D 역투영이지만, 3D 상태공간에서
   직접 칼만필터를 돌리면 다중 카메라 관측을 자연스럽게 융합할 수 있음.
4. **정확한 per-scene calibration 3D backprojection**: 현재는 지면(Z=0) 가정만 사용;
   `cameraMatrix`(full 3D)와 다중 카메라 삼각측량을 결합하면 Z 추정도 가능.
5. **motmetrics/HOTA 자체 평가 스크립트**: `ground_truth.json`을 이용해 `track1.txt`의
   MOTA/HOTA를 로컬에서 계산하는 스크립트 추가 (submission 전 자체 검증용).

## 6. 알려진 블로커 / 미완료 부분

- Track1 스펙 문서에 언급된 `Warehouse_004`(당시 7/N개 비디오만 존재)는 이번 작업 시점에는
  전체 다운로드가 완료되어 19개 비디오 + `ground_truth.json`이 모두 존재함 (재확인 완료,
  아래 표 참조). 데이터 다운로드는 계속 진행 중인 것으로 보이며(`train/`에 Warehouse_000~012,
  `test/`에 Warehouse_023~027 확인), 본 작업은 스모크테스트에 필요한 Warehouse_000 5개
  카메라·250 프레임만 사용했고 나머지는 손대지 않음.
- `scene_id` 매핑: 데이터셋 내에 별도의 scene-id 매핑 파일이 없어서, `Warehouse_XXX` 폴더명의
  숫자 접미사를 그대로 `scene_id`로 사용함 (`export_submission.py::scene_to_id`). 실제 제출
  규격과 다를 경우 이 함수만 수정하면 됨.
- `test/` 셋(Warehouse_023-027, `ground_truth.json` 없음 — 예상대로 정상)에 대해서는 파이프라인
  코드는 동일하게 동작하도록 작성했으나 실제 실행/검증은 하지 않음 (train 세트로 정확도
  검증 우선).
- YOLO 모델 가중치 `yolo11n.pt` (5.4MB)가 `ultralytics`에 의해 이 디렉토리에 자동 다운로드됨.

### 로컬 데이터 현황 (본 작업 시점 재확인)
| scene | videos | ground_truth.json |
|---|---|---|
| Warehouse_000 | 19 | O |
| Warehouse_001 | 19 | O |
| Warehouse_002 | 20 | O |
| Warehouse_003 | 20 | O |
| Warehouse_004 | 19 | O |
| Warehouse_005 | 19 | O |
| Warehouse_006 | 13 | O |
| test/Warehouse_023..027 | 4~20 | X (정상, 실제 테스트셋) |

## 7. 검증 결과 (Warehouse_000, 5개 카메라, 250프레임 스모크테스트)

- 카메라: `Camera_0000, Camera_0003, Camera_0007, Camera_0011, Camera_0015`
- YOLO11n 추론 속도: RTX 3090에서 약 30-40 fps/카메라 (COCO pretrained, conf=0.25)
- 결과: `track1.txt` 1581줄, 11개 필드 전부 파싱 성공(malformed=0), class_id ∈ {0,1},
  object_id 45개, frame_id 250개, 좌표 범위 x:[-91.80,-7.65] y:[-95.65,-13.14] z:[0.92,1.29]
  (전부 미터 스케일의 정상적인 값, NaN/inf 없음)
- **ground_truth.json 대비 좌표 정확도 (frame 0, Person)**:
  - pred(-84.18,-90.20) vs GT object 488 (-83.90,-90.23) → 거리 0.28m
  - pred(-9.48,-51.73) vs GT object 2424 (-9.67,-51.77) → 거리 0.19m
  - pred(-8.67,-57.63) vs GT object 2418 (-8.81,-57.63) → 거리 0.14m
  - pred(-10.99,-56.04) vs GT object 2410 (-11.15,-56.08) → 거리 0.16m
  - (frame 3) Forklift 프록시 pred(-71.08,-13.46) vs GT object 4486 Forklift (-71.58,-13.35)
    → 거리 0.52m (클래스는 프록시로 우연히 맞았으나 일반화되지 않음, 4절 한계점 참조)
  → **지면 호모그래피 역투영이 실제 3D 위치를 0.2-0.5m 오차 내로 잘 복원함을 확인**.
  ID(object_id)는 GT와 매칭하지 않음(스펙상 요구되지 않음, 위치 정확도만 확인).

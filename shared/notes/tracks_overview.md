# AI City Challenge 2026 — 트랙 개요 (조사일: 2026-07-02)

마감: 2026-07-10. 평가시스템: https://eval.aicitychallenge.org/aicity2026/login (기관 이메일 필수, 승인 24-48h)
제출 제한: 트랙당 일 5회 / 전체 20회. 문의: aicitychallenges@gmail.com

## Track 1 — Multi-Camera 3D Perception (Sim2Real)
- 태스크: 다중 카메라 간 동일 ID 유지 추적 (사람/AMR/휴머노이드/지게차), RGB만 추론(depth는 학습시만)
- 데이터: hf nvidia/PhysicalAI-SmartSpaces (train/val), 28.5h, 342 카메라, ~50GB(depth 제외)/~3TB(포함)
- 평가: 3D HOTA (+10% 온라인추적 보너스, 논문/코드 증명 필요)
- 제출: track1.txt, `<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <w> <l> <h> <yaw>`, 50MB 이하

## Track 2 — Transportation Safety Captioning + VQA (Sim2Real)
- 태스크: (1) 보행자/차량 각각 캡션 생성 (2) ~180유형 객관식 VQA
- 데이터: hf mlcglab/synwts(합성, 학습만 가능) / github woven-visionai/wts-dataset(실제, 테스트만)
- 주의: 실제 WTS 학습데이터·사전학습모델 사용 금지, 생성형 필수(검색기반 금지)
- 평가: (BLEU-4+METEOR+ROUGE-L+CIDEr)/4 와 VQA정확도의 평균
- 제출: JSON (스키마 fetch 결과 참조)

## Track 3 — Anomalous Events in Transportation
- 태스크: 10개 서브태스크(이벤트검증/설명/MCQ/오픈QA/장면요약/원인분석 등), temporal localization은 채점 제외(26-07-01 공지)
- 데이터: hf nvidia/PhysicalAI-Traffic-Anomaly-Reasoning, 44,040 주석/3,670영상(~26.1h), 8개 소스에서 영상 스크립트로 수집
- 평가: 3개 독립 리더보드 - TAR(인도메인, bcq/mcq 정확도+BERTScore F1 평균), FETV(OOD1), PSI VQA(OOD2, 비상업 학술용도만)
- 제출: CSV(item_index,prediction) 또는 JSON(FETV)

## Track 4 — Text-Based Person Anomaly Search (Sim2Real)
- 태스크: 자연어로 이상행동(낙상/폭력피해 등) 사람 검색, PAB 벤치마크
- 데이터: github Shuyu-XJTU/PAB-for-ECCV26-Workshop-Track4, 학습 1,013,605 합성이미지+텍스트, 테스트 쿼리1978/갤러리1978+방해34795
- 평가: mAP
- 제출: answer.txt, 쿼리당 top-10 이미지ID
- 주의: 테스트 분포 학습/검증/임계값조정에 절대 사용 금지, 수상시 코드 제출 의무

## Track 5 — Generative Traffic Video Forecasting
- 태스크: 캡션2개+초기 프레임 조건으로 미래 프레임 생성
- 데이터: **구글폼 신청 필요** https://forms.gle/szQPk1TMR8JXzm327 (승인후 이메일), 810영상/155시나리오, BDD_PC_5K 사전학습 활용 가능
- 평가: PSNR/SSIM/LPIPS/CLIP-S/FVD 평균
- 제출: 0.png ~ N-1.png 시퀀스, 입력과 동일 해상도
- 주의: WTS 테스트데이터 학습금지, 사전학습/외부데이터 사용시 리포트에 명시 필수

## Track 6 — Cross-City Object Detection
- 태스크: 단일이미지 10클래스 탐지(차량세분류+사람), 지리적 도메인시프트 일반화
- 데이터: **직접 다운로드 불가**, Hafnia Training-as-a-Service 플랫폼 통해서만 (코드/도커 업로드 → 플랫폼에서 학습)
- 제약: 참가자당 30,000 크레딧, 동시 실험 1개만, 업로드 2GB 이하, 외부데이터 금지, 별도 Hafnia 계정+등록 필요(선착순 200팀)
- 평가: mAP (클래스/도시별)
- 별도 등록: https://community.hafnia.milestonesys.com/home/clubs/ai-city-challenge-track-6-omnhs/overview

## 공통 리스크
- Track1 데이터 3TB(depth 포함)는 디스크 무리 → depth 제외 버전(~50GB)만 사용
- Track4 학습셋 100만+ 이미지, 다운로드/스토리지 시간 확인 필요
- Track5는 승인 대기 시간이 변수 (구글폼 → 이메일 전송, 기간 불명)
- Track6은 로컬 GPU 불필요하지만 별도 플랫폼 학습이라 워크플로우 완전히 다름, 30,000 크레딧 소진 관리 필요

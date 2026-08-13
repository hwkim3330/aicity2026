# Dataset and model data policy

Challenge data and model weights are **not included** in this repository.

FETV and PSI-VQA are public and can be fetched directly:

- **FETV** — 200 clips, 563 MiB. [github.com/MoyoG/FETV](https://github.com/MoyoG/FETV)
  links a Google Drive folder; `python3 -m gdown --folder <url>` then unzip.
- **PSI-VQA** — `hf download ise-ice-lab/PSI_VQA --repo-type dataset --include "test_public/*"`.

TAR test clips come through the official AI City Challenge organizer
distribution. Place everything under the paths documented in each track
README. Do not redistribute test clips, annotations,
or private evaluation data. Follow the individual FETV, PSI-VQA, TAR, WTS,
and AI City Challenge licenses and terms. Download `Qwen/Qwen3-VL-8B-Instruct`
from Hugging Face under its model license; weights remain outside Git.

Expected examples:

```text
track3_anomaly/data/fetv/FETV_public_clips/*.mp4
track3_anomaly/data/psi_vqa/{train,test_public}/
track3_anomaly/data/test/test.json
```

Never commit credentials, gated-data URLs, private test data, or model cache
directories.

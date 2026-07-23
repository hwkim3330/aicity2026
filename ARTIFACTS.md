# Artifact Manifest

Large generated outputs remain local to avoid bloating normal Git history.

| Artifact | Size | SHA256 |
|---|---:|---|
| `track1_3dperception/track1_test_v6_boundfix.txt` | 35,250,758 bytes | `8eddf764d8991722d4c03c82c6a65e151be7dd0188932f2147e45b39844aab50` |
| `track6_detection/submissions/track6_rfdetr_v5_7ep.json` | 64,524,638 bytes | `647ed56519bb1cb9754ed6faa76f462280a87446a8121f53b36531704e93fdce` |
| `track3_anomaly/submissions/fetv_submission_v11.json` | 200 JSON records; 3,202 physical lines | `39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835` |
| `track3_anomaly/submissions/psi_vqa_submission_v7.csv` | 328 prediction records plus one header; 455 physical lines because quoted predictions contain embedded line breaks | `a3829a36f591907bb8838098b1cc61feb907fec1cc6215f6098094aafaafb110` |

Small Track 7/8 final artifacts are versioned under
`track3_anomaly/submissions/`. Model weights, datasets, caches, and virtual
environments are intentionally excluded.

`psi_vqa_submission.csv` is the earlier 328-item candidate retained for history.
The final portal upload filename was not captured in the repository; v7 is the
repository-side candidate associated with the final 57.0400 result. The
post-deadline research file `psi_vqa_submission_v8_final.csv` is intentionally
not listed as official; see its notice file.

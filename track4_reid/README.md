# Track 4 — Text-Based Person Anomaly Search (PAB)

Natural-language query -> retrieve matching images from a mixed gallery
(1978 true matches + 34795 distractors). Ranked by mAP; submission is
`answer.txt`, one line per query (same order as `data/query_index.txt`),
each line the top-10 gallery image IDs.

## Data (already downloaded, in `data/`)

- `data/gallery/*.jpg` — 36,773 images (query matches + distractors), masked filenames.
- `data/query_text.json` — JSONL, one `{query_index, caption, change}` per line.
- `data/query_index.txt` — submission row order.

`train_webp.zip` (1M+ captioned images, for fine-tuning) is still
downloading separately — not needed for the zero-shot baseline below, only
for improving on it later.

## Baseline: zero-shot CLIP retrieval

No training needed to get a first submission:

```bash
./run_pipeline.sh          # embeds gallery once (~cached to data/gallery_embeds.pt), then retrieves
python3 validate_submission.py --answer answer.txt
cp answer.txt repo/answer.txt   # repo/ is the upstream PAB repo clone, has the submission example
```

- `common.py` — CLIP (`openai/clip-vit-large-patch14`) load/embed helpers.
- `embed_gallery.py` — embeds all 36.7k gallery images once, caches to disk.
- `retrieve.py` — embeds query captions, cosine-similarity ranks against
  the cached gallery embeddings, writes `answer.txt`.
- `validate_submission.py` — checks row count, top-k count, no duplicate
  IDs, all IDs exist in the gallery — catches format bugs before you spend
  one of the 5/day submission slots.

## Next steps (once `train_webp.zip` finishes downloading)

The captions in `annotation/train/imgs_*.json` (13.6k lines per shard) look
like: `{"image": "...", "caption": "...", "image_id": "...", "scene": "...",
"normal": "Performing"}` — i.e. domain captions describing both normal and
anomalous human actions, matching the query style (falls, being carried,
tumbling off a bicycle, etc.). Fine-tuning CLIP contrastively on this
in-domain data (image, caption) pairs should meaningfully beat the
zero-shot baseline, since generic CLIP wasn't trained on "person falling
off bicycle" style anomaly descriptions. Rule check: PAB's own README says
nothing forbids using the released *training* set (only the *test*
distribution is off-limits for tuning/thresholds) — reconfirm against the
official rules page before relying on this.

## Tried and abandoned: Qwen3-VL-Embedding

Benchmarks looked promising (0.945 vs CLIP's 0.768 on hard cross-modal retrieval,
per arxiv 2601.04720), but the sentence-transformers integration for
`Qwen/Qwen3-VL-Embedding-2B` fails to load checkpoint weights (every
language_model/visual layer comes back "newly initialized" -- confirmed
reproducible across two transformers versions, 4.57.0 and the model's own
required 4.57.1). Retrieval output is unrelated to query content, consistent
with random weights. Not pursued further given the time budget; the CLIP
pipeline (answer_rerank.txt / answer_ensemble.txt) stands.

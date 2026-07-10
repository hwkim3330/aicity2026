#!/usr/bin/env python3
"""
logdump.py -- exfiltrate a file through the Hafnia stdout log channel.

WHY THIS EXISTS
    Hafnia has no artifact download: no checkpoint retrieval, no output-file
    retrieval. The ONLY channel confirmed retrievable after a job ends is
    GET /api/v1/experiments/{id}/logs  (stdout/stderr entries), which is
    empirically capped at ~1000 entries per request. So the submission JSON
    must be printed to stdout, compressed + base64-encoded, in self-describing
    chunks that a local script (scripts/logdump_client.py in the repo root,
    NOT in the trainer package) can find and reassemble even if some entries
    are lost, reordered, or truncated.

WIRE FORMAT (one chunk per print(), strictly single-line ASCII)
    Header/footer (printed twice before chunk 1 and twice after the last):
      @HAFDUMP@|H|<tag>|chunks=<N>|chunk_chars=<C>|codec=<gzip|lzma>|enc=b64|raw_bytes=<n>|raw_md5=<hex>|comp_bytes=<n>|enc_chars=<n>|enc_md5=<hex>|@END@
    Chunk i (1-based):
      @HAFDUMP@|C|<tag>|<i>/<N>|<payload_len>:<crc32_hex>|<payload>|@END@

    raw_md5  = md5 of the original (pre-compression) payload bytes
    enc_md5  = md5 of the FULL base64 string (concatenation of all chunks)
    Per-chunk crc32 + explicit length + trailing @END@ make silent per-line
    truncation detectable; base64-only payload means no char ever needs
    escaping by any log pipeline. (base85 is deliberately NOT supported: its
    alphabet contains '|' and '@', which would break these delimiters.)

CHUNK SIZE
    DEFAULT_CHUNK_CHARS = 4000 is deliberately conservative: cloud log
    pipelines truncate single events anywhere from ~4KB to ~256KB and we do
    not yet know Hafnia's limit. Measured payload budget (synthetic 248,660
    detections / 14,924 images, worst-case-random content):
        compact payload, lzma, base64  ->  ~4.1-5.0 MB encoded
        @4000 chars/chunk  -> ~1030-1250 chunks  (OVER the ~1000-entry cap!)
        @6000 chars/chunk  ->  ~680-840 chunks   (ok)
        @8000 chars/chunk  ->  ~510-630 chunks   (comfortable)
    => Run scripts/log_probe.py as a cheap experiment FIRST; if lines of
    8000+ chars survive intact, launch the real run with
    --logdump-chunk-chars 8000. 4000 is only safe if pagination past the
    1000-entry cap turns out to work (the probe + local client test that too).
"""
import base64
import binascii
import gzip
import hashlib
import json
import lzma
import sys
import time

MARK = "@HAFDUMP@"
ENDMARK = "@END@"
DEFAULT_CHUNK_CHARS = 4000   # chars of base64 per print(); see module docstring
DEFAULT_CODEC = "lzma"       # lzma is ~20% smaller than gzip on this payload
DEFAULT_SLEEP_MS = 15        # pause between prints so a rate-limited log
                             # collector doesn't drop bursts (700 chunks ~ 10s)

COMPACT_FORMAT_NAME = "aicity-track6-compact-v1"


def _crc(s: str) -> str:
    return format(binascii.crc32(s.encode("ascii")) & 0xFFFFFFFF, "08x")


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def compress(raw: bytes, codec: str) -> bytes:
    if codec == "gzip":
        return gzip.compress(raw, 9)
    if codec == "lzma":
        return lzma.compress(raw, preset=9)
    raise ValueError(f"unknown codec {codec!r}")


def decompress(comp: bytes, codec: str) -> bytes:
    if codec == "gzip":
        return gzip.decompress(comp)
    if codec == "lzma":
        return lzma.decompress(comp)
    raise ValueError(f"unknown codec {codec!r}")


def dump_bytes(raw: bytes, tag: str,
               chunk_chars: int = DEFAULT_CHUNK_CHARS,
               codec: str = DEFAULT_CODEC,
               sleep_ms: int = DEFAULT_SLEEP_MS,
               stream=None) -> int:
    """Compress+encode raw bytes and print them as self-describing chunks.
    Returns the number of chunks emitted. Never raises for I/O reasons the
    caller cares about -- but callers should still wrap in try/except so a
    dump bug can never kill a 21-hour job at the finish line."""
    stream = stream or sys.stdout
    comp = compress(raw, codec)
    enc = base64.b64encode(comp).decode("ascii")
    n = max(1, -(-len(enc) // chunk_chars))
    header = (f"{MARK}|H|{tag}|chunks={n}|chunk_chars={chunk_chars}"
              f"|codec={codec}|enc=b64|raw_bytes={len(raw)}|raw_md5={_md5(raw)}"
              f"|comp_bytes={len(comp)}|enc_chars={len(enc)}"
              f"|enc_md5={hashlib.md5(enc.encode('ascii')).hexdigest()}|{ENDMARK}")
    sleep = max(0, sleep_ms) / 1000.0

    for _ in range(2):                      # header twice: survives one loss
        print(header, file=stream, flush=True)
        time.sleep(sleep)
    for i in range(n):
        payload = enc[i * chunk_chars:(i + 1) * chunk_chars]
        print(f"{MARK}|C|{tag}|{i + 1}/{n}|{len(payload)}:{_crc(payload)}"
              f"|{payload}|{ENDMARK}", file=stream, flush=True)
        time.sleep(sleep)
    for _ in range(2):                      # footer twice
        print(header, file=stream, flush=True)
        time.sleep(sleep)
    return n


def dump_file(path, tag: str, **kw) -> int:
    with open(path, "rb") as f:
        return dump_bytes(f.read(), tag, **kw)


# ---------------------------------------------------------------------------
# Compact submission payload: ~5x smaller raw than the flat record list that
# benchmark.py writes to disk, and losslessly reconstructable back to it
# (up to the reduced float precision chosen at dump time -- 0.05 px / 0.0005
# score has no measurable effect on mAP).
# ---------------------------------------------------------------------------
def compact_payload(fused, id_of, class_names,
                    bbox_decimals: int = 1, score_decimals: int = 3) -> bytes:
    """fused: {image_stem: [{'category_id','bbox','score'}, ...]}  (the exact
    structure benchmark.py builds); id_of: {stem: {'file_path','sample_index',...}}
    or None; class_names: list[str]."""
    images = {}
    for stem, boxes in fused.items():
        images[stem] = [
            [int(b["category_id"]),
             *[round(float(v), bbox_decimals) for v in b["bbox"]],
             round(float(b["score"]), score_decimals)]
            for b in boxes
        ]
    meta = {}
    for stem in fused:
        m = (id_of or {}).get(stem, {})
        meta[stem] = [m.get("file_path"), m.get("sample_index")]
    obj = {"format": COMPACT_FORMAT_NAME,
           "class_names": list(class_names),
           "meta": meta,
           "images": images}
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def reconstruct_records(payload: dict) -> list:
    """Inverse of compact_payload: rebuild the flat record list in exactly the
    schema benchmark.py writes to submission.json. Used LOCALLY by
    scripts/logdump_client.py after reassembly."""
    assert payload.get("format") == COMPACT_FORMAT_NAME, payload.get("format")
    class_names = payload["class_names"]
    meta = payload.get("meta", {})
    records = []
    for stem, dets in payload["images"].items():
        fp, si = (meta.get(stem) or [None, None])[:2]
        for cid, x, y, w, h, score in dets:
            cid = int(cid)
            records.append({
                "image_id": stem,
                "file_path": fp,
                "sample_index": si,
                "category_id": cid,
                "category_name": class_names[cid] if cid < len(class_names) else str(cid),
                "bbox": [x, y, w, h],
                "score": score,
            })
    return records


if __name__ == "__main__":
    # standalone: python scripts/logdump.py <file> [tag] [chunk_chars]
    path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "file"
    cc = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CHUNK_CHARS
    n = dump_file(path, tag, chunk_chars=cc)
    print(f"[logdump] emitted {n} chunks for {path}", file=sys.stderr)

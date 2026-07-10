#!/usr/bin/env python3
"""
logdump_client.py -- local companion to trainer_package/scripts/logdump.py.

Hafnia exposes no artifact download; the only retrievable output channel is
GET /api/v1/experiments/{id}/logs (empirically capped at ~1000 entries per
request). This tool (a) fetches those logs trying several pagination
strategies and reports which, if any, get past the cap; (b) analyzes a
scripts/log_probe.py run; (c) reassembles a submission dumped by
logdump.dump_bytes() back into submission.json, verifying every chunk crc
and the whole-payload md5 before writing anything.

Usage:
  python scripts/logdump_client.py fetch <experiment_id> [-o logs.json]
  python scripts/logdump_client.py probe-report logs.json [more_logs.json ...]
  python scripts/logdump_client.py reassemble logs.json [more.json ...] \
      [--tag submission] [-o submission.json] [--raw-payload-out compact.json]
  python scripts/logdump_client.py selftest

Auth comes from ~/.hafnia/config.json (active profile: platform_url + api_key,
header "Authorization: <api_key>" -- same convention as the hafnia SDK).
`fetch` may be run several times / with several strategies; `reassemble` and
`probe-report` accept multiple log files and merge them, so partial fetches
add up.
"""
import argparse
import base64
import binascii
import hashlib
import importlib.util
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGDUMP_PY = REPO_ROOT / "trainer_package" / "scripts" / "logdump.py"

CHUNK_RE = re.compile(
    r"@HAFDUMP@\|C\|(?P<tag>[\w.-]+)\|(?P<idx>\d+)/(?P<total>\d+)"
    r"\|(?P<len>\d+):(?P<crc>[0-9a-f]{8})\|(?P<payload>[A-Za-z0-9+/=]*)")
HEADER_RE = re.compile(
    r"@HAFDUMP@\|H\|(?P<tag>[\w.-]+)\|chunks=(?P<chunks>\d+)"
    r"\|chunk_chars=(?P<chunk_chars>\d+)\|codec=(?P<codec>\w+)\|enc=b64"
    r"\|raw_bytes=(?P<raw_bytes>\d+)\|raw_md5=(?P<raw_md5>[0-9a-f]{32})"
    r"\|comp_bytes=(?P<comp_bytes>\d+)\|enc_chars=(?P<enc_chars>\d+)"
    r"\|enc_md5=(?P<enc_md5>[0-9a-f]{32})")
PROBE_SIZE_RE = re.compile(
    r"@HAFPROBE@\|SIZE\|(?P<phase>\w+)\|(?P<stream>\w+)\|(?P<size>\d+)"
    r"\|(?P<len>\d+):(?P<crc>[0-9a-f]{8})\|(?P<payload>[A-Za-z0-9+/]*)")
PROBE_SEQ_RE = re.compile(r"@HAFPROBE@\|(?P<kind>BURST|THROT)\|(?P<i>\d+)/(?P<n>\d+)\|")


def load_logdump_module():
    spec = importlib.util.spec_from_file_location("logdump", LOGDUMP_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crc32_hex(s: str) -> str:
    return format(binascii.crc32(s.encode("ascii")) & 0xFFFFFFFF, "08x")


# ---------------------------------------------------------------- fetch ----
def hafnia_auth():
    cfg = json.loads((Path.home() / ".hafnia" / "config.json").read_text())
    prof = cfg["profiles"][cfg["active_profile"]]
    return prof["platform_url"].rstrip("/"), prof["api_key"]


def api_get(url: str, api_key: str, params: dict):
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(f"{url}?{q}" if q else url,
                                 headers={"Authorization": api_key,
                                          "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def extract_entries(resp):
    """Return the list of log entries from whatever envelope the API uses."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for key in ("results", "data", "items", "logs", "entries"):
            if isinstance(resp.get(key), list):
                return resp[key]
    return [resp]


def entry_key(e) -> str:
    return json.dumps(e, sort_keys=True, ensure_ascii=True)


def entry_timestamp(e):
    if isinstance(e, dict):
        for k in ("created_at", "timestamp", "time", "ts"):
            if k in e:
                return e[k]
    return None


def cmd_fetch(args):
    base_url, api_key = hafnia_auth()
    ep = f"{base_url}/api/v1/experiments/{args.experiment_id}/logs"
    seen = {}
    report = []

    def grab(name, params):
        try:
            resp = api_get(ep, api_key, params)
        except Exception as e:  # noqa: BLE001
            report.append((name, f"ERROR {e}"))
            return 0
        entries = extract_entries(resp)
        new = 0
        for e in entries:
            k = entry_key(e)
            if k not in seen:
                seen[k] = e
                new += 1
        report.append((name, f"{len(entries)} returned, {new} new (total {len(seen)})"))
        return new

    # baseline both orderings -- if the cap keeps one end of the stream,
    # asking for both ends may already double the budget to ~2000 entries.
    grab("newest-first", {"limit": args.limit, "ordering": "-created_at"})
    grab("oldest-first", {"limit": args.limit, "ordering": "created_at"})

    # offset/page pagination attempts
    for pname, pval in (("offset", args.limit), ("page", 2), ("skip", args.limit)):
        grab(f"pagination {pname}={pval}",
             {"limit": args.limit, "ordering": "-created_at", pname: pval})

    # time-window pagination: walk backwards from the oldest timestamp we have
    ts = [t for t in (entry_timestamp(e) for e in seen.values()) if t]
    if ts:
        oldest = min(ts)
        for pname in ("created_at__lt", "before", "until", "created_at_lt"):
            n = grab(f"time-window {pname}<{oldest}",
                     {"limit": args.limit, "ordering": "-created_at", pname: oldest})
            if n:
                # this strategy works -- keep walking back until exhausted
                while n:
                    ts = [t for t in (entry_timestamp(e) for e in seen.values()) if t]
                    n = grab(f"time-window {pname}<{min(ts)} (cont)",
                             {"limit": args.limit, "ordering": "-created_at",
                              pname: min(ts)})
                break

    out = Path(args.out)
    out.write_text(json.dumps(list(seen.values()), indent=1))
    print(f"\n=== fetch report for experiment {args.experiment_id} ===")
    for name, res in report:
        print(f"  {name:40s} {res}")
    print(f"  -> {len(seen)} unique entries saved to {out}")
    if len(seen) <= args.limit and len(seen) >= 900:
        print("  NOTE: entry count sits near the ~1000 cap and no strategy "
              "broke past it -> assume only ~1000 (likely newest) entries are "
              "retrievable; size the dump accordingly (chunk_chars >= 8000).")


# ---------------------------------------------------------- text mining ----
def all_text(log_files):
    """Concatenate every string found in the given files (JSON or raw text);
    chunk lines are self-describing so we can regex the whole blob."""
    parts = []
    for f in log_files:
        raw = Path(f).read_text(errors="replace")
        try:
            obj = json.loads(raw)
        except ValueError:
            parts.append(raw)
            continue
        def walk(o):
            if isinstance(o, str):
                parts.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(obj)
    return "\n".join(parts)


# ---------------------------------------------------------- probe-report ---
def cmd_probe_report(args):
    text = all_text(args.log_files)

    print("=== SIZE ladder (largest intact single log line) ===")
    best_intact = 0
    for m in PROBE_SIZE_RE.finditer(text):
        size, plen = int(m["size"]), int(m["len"])
        payload = m["payload"]
        intact = len(payload) == plen and crc32_hex(payload) == m["crc"]
        status = "INTACT" if intact else f"TRUNCATED at {len(payload)} chars"
        print(f"  {m['phase']:5s} {m['stream']:6s} {size:>7d} chars: {status}")
        if intact:
            best_intact = max(best_intact, size)
    print(f"  -> largest intact line: {best_intact} chars"
          if best_intact else "  -> NO size lines found at all")

    print("\n=== BURST / THROTTLE survival (entry cap & rate drops) ===")
    for kind in ("BURST", "THROT"):
        got = sorted({int(m["i"]) for m in PROBE_SEQ_RE.finditer(text)
                      if m["kind"] == kind})
        if not got:
            print(f"  {kind}: 0 lines retrieved")
            continue
        total = max(int(m["n"]) for m in PROBE_SEQ_RE.finditer(text)
                    if m["kind"] == kind)
        # summarize contiguous ranges to show WHICH part survived
        ranges, start = [], got[0]
        for a, b in zip(got, got[1:]):
            if b != a + 1:
                ranges.append((start, a)); start = b
        ranges.append((start, got[-1]))
        rtxt = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges[:10])
        print(f"  {kind}: {len(got)}/{total} retrieved; ranges: {rtxt}"
              + (" ..." if len(ranges) > 10 else ""))

    print("\n=== REALDUMP rehearsal (production logdump path, tag 'probe') ===")
    ok = try_reassemble(text, "probe", out_path=None, quiet=False)
    print("  -> production dump/reassemble path: " + ("VERIFIED OK" if ok else "FAILED"))

    print("\nDecision guide:")
    print("  chunk budget = (#entries retrievable) - (~50 margin for other logs)")
    print("  needed chunks ~= 5.0 MB base64 / chunk_chars   (conservative)")
    print("  pick the largest chunk_chars that was INTACT above; pass it to the")
    print("  real run as --logdump-chunk-chars. 8000+ intact => ~630 chunks => safe.")


# ----------------------------------------------------------- reassemble ----
def try_reassemble(text, tag, out_path, quiet=False,
                   raw_payload_out=None, write_records=True):
    ld = load_logdump_module()
    headers = [m.groupdict() for m in HEADER_RE.finditer(text)
               if m["tag"] == tag]
    chunks = {}
    bad = 0
    for m in CHUNK_RE.finditer(text):
        if m["tag"] != tag:
            continue
        idx, payload = int(m["idx"]), m["payload"]
        if len(payload) != int(m["len"]) or crc32_hex(payload) != m["crc"]:
            bad += 1
            continue  # truncated/corrupt copy; a duplicate may still be good
        chunks[idx] = payload

    if not headers:
        if not quiet:
            print(f"  no @HAFDUMP@ header for tag {tag!r} found")
        return False
    h = headers[-1]
    total = int(h["chunks"])
    missing = sorted(set(range(1, total + 1)) - set(chunks))
    if not quiet:
        print(f"  header: {total} chunks x {h['chunk_chars']} chars, codec={h['codec']}, "
              f"raw {h['raw_bytes']} B, enc {h['enc_chars']} chars")
        print(f"  chunks recovered: {len(chunks)}/{total}"
              f" ({bad} corrupt/truncated copies discarded)")
    if missing:
        mr = ", ".join(map(str, missing[:20]))
        print(f"  MISSING chunk indices ({len(missing)}): {mr}"
              + (" ..." if len(missing) > 20 else ""))
        return False

    enc = "".join(chunks[i] for i in range(1, total + 1))
    if len(enc) != int(h["enc_chars"]) or \
            hashlib.md5(enc.encode("ascii")).hexdigest() != h["enc_md5"]:
        print("  FULL-PAYLOAD md5 MISMATCH after reassembly -- refusing to decode")
        return False
    raw = ld.decompress(base64.b64decode(enc), h["codec"])
    if len(raw) != int(h["raw_bytes"]) or hashlib.md5(raw).hexdigest() != h["raw_md5"]:
        print("  RAW md5 mismatch after decompress -- corrupt")
        return False
    if not quiet:
        print(f"  md5 verified: enc={h['enc_md5']} raw={h['raw_md5']}")

    if raw_payload_out:
        Path(raw_payload_out).write_bytes(raw)
        print(f"  raw compact payload -> {raw_payload_out}")
    if out_path:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if write_records and isinstance(payload, dict) \
                and payload.get("format") == ld.COMPACT_FORMAT_NAME:
            records = ld.reconstruct_records(payload)
            Path(out_path).write_text(json.dumps(records))
            n_img = len(payload["images"])
            print(f"  reconstructed {len(records)} detections across {n_img} "
                  f"images -> {out_path}")
        else:
            Path(out_path).write_bytes(raw)
            print(f"  payload (non-compact format) -> {out_path}")
    return True


def cmd_reassemble(args):
    text = all_text(args.log_files)
    ok = try_reassemble(text, args.tag, args.out,
                        raw_payload_out=args.raw_payload_out)
    if not ok:
        print("\nreassembly INCOMPLETE. Options: re-run `fetch` (dup entries "
              "merge for free), fetch with other strategies, pass multiple "
              "log files here.")
        sys.exit(1)


# -------------------------------------------------------------- selftest ---
def cmd_selftest(_args):
    """Round-trip WITHOUT the network: synthetic detections -> compact payload
    -> dump_bytes into a buffer -> (simulated entry shuffling + one duplicated
    truncated chunk) -> reassemble -> byte-identical record check."""
    import io
    import random
    ld = load_logdump_module()
    random.seed(1)
    fused, id_of = {}, {}
    for i in range(400):
        stem = f"img{i:05d}" + format(random.getrandbits(48), "012x")
        id_of[stem] = {"file_path": f"data/{stem}.jpg", "sample_index": i}
        fused[stem] = [{
            "category_id": random.randrange(10),
            "bbox": [round(random.uniform(0, 1500), 2), round(random.uniform(0, 900), 2),
                     round(random.uniform(8, 400), 2), round(random.uniform(8, 300), 2)],
            "score": round(random.uniform(0.15, 0.99), 4),
        } for _ in range(random.randint(1, 30))]
    class_names = [f"class_{c}" for c in range(10)]
    payload = ld.compact_payload(fused, id_of, class_names)

    buf = io.StringIO()
    n = ld.dump_bytes(payload, tag="submission", chunk_chars=4000,
                      sleep_ms=0, stream=buf)
    lines = buf.getvalue().splitlines()
    print(f"dumped: {len(payload)} raw bytes -> {n} chunks, {len(lines)} log lines")

    # simulate a hostile log pipeline: shuffle entries, wrap them in a JSON
    # envelope like the API would, truncate one DUPLICATED chunk copy
    dup = next(l for l in lines if "|C|submission|17/" in l)
    lines.append(dup[:len(dup) // 2])          # truncated duplicate
    random.shuffle(lines)
    entries = [{"created_at": f"t{i:06d}", "message": l}
               for i, l in enumerate(lines)]
    logs_file = Path("/tmp/logdump_selftest_logs.json")
    logs_file.write_text(json.dumps({"results": entries}))

    out = Path("/tmp/logdump_selftest_submission.json")
    ok = try_reassemble(all_text([logs_file]), "submission", out)
    assert ok, "reassembly failed"

    got = json.loads(out.read_text())
    want = ld.reconstruct_records(json.loads(payload.decode()))
    assert got == want, "reconstructed records differ from expected"
    n_det = sum(len(v) for v in fused.values())
    assert len(got) == n_det, (len(got), n_det)
    # spot-check reduced-precision fidelity vs the original fused boxes
    by_img = {}
    for r in got:
        by_img.setdefault(r["image_id"], []).append(r)
    stem = next(iter(fused))
    for orig, rec in zip(fused[stem], by_img[stem]):
        assert rec["category_id"] == orig["category_id"]
        assert all(abs(a - b) <= 0.05 + 1e-9 for a, b in zip(rec["bbox"], orig["bbox"]))
        assert abs(rec["score"] - orig["score"]) <= 0.0005 + 1e-9
    print(f"SELFTEST PASSED: {len(got)} records round-tripped byte-exactly "
          f"(incl. shuffled entries + truncated duplicate chunk)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="fetch experiment logs, probing pagination")
    f.add_argument("experiment_id")
    f.add_argument("-o", "--out", default="logs.json")
    f.add_argument("--limit", type=int, default=5000)
    f.set_defaults(fn=cmd_fetch)

    pr = sub.add_parser("probe-report", help="analyze a log_probe.py run")
    pr.add_argument("log_files", nargs="+")
    pr.set_defaults(fn=cmd_probe_report)

    r = sub.add_parser("reassemble", help="rebuild submission.json from logs")
    r.add_argument("log_files", nargs="+")
    r.add_argument("--tag", default="submission")
    r.add_argument("-o", "--out", default="submission.json")
    r.add_argument("--raw-payload-out", default=None,
                   help="also save the raw compact payload JSON")
    r.set_defaults(fn=cmd_reassemble)

    s = sub.add_parser("selftest", help="offline round-trip test")
    s.set_defaults(fn=cmd_selftest)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

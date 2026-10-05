"""Verify the pre-registration locks of this repository before (re)running anything.

Checks (always):
  - configs/protocol_v5.json against configs/protocol_v5_lock.json (SHA-256)
  - the four addenda (E, E3b, R4, F16) against their lock files and the base protocol hash
  - the locked code files listed in protocol_v5.json["code"] (the V3/V4 scripts repeat this check and stop on mismatch)
  - reports/analysis/v5_image_manifest.json against the lock
  - the protocol hash recorded in the V3 thresholds and in the V4 summary
Optional:
  --wam  PATH   WAM repository: commit, locked source files and checkpoints (protocol_v5.json["wam"]["files_sha256"])
  --coco PATH   COCO val2017 directory: the four per-split image-list hashes of the lock

Usage:  python scripts/check_locks.py [--wam /workspace/wam/JISA_selected_sources/watermark-anything]
                                      [--coco /workspace/wam/datasets/coco2017/val2017]
Exit code 0 if every performed check passes, 1 otherwise.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def J(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wam", type=Path, help="WAM repository checked out at the locked commit")
    ap.add_argument("--coco", type=Path, help="COCO val2017 image directory")
    args = ap.parse_args()

    lock, proto = J("configs/protocol_v5_lock.json"), J("configs/protocol_v5.json")
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"[{'OK' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    check("protocol_v5.json", sha(ROOT / lock["protocol"]) == lock["protocol_sha256"], lock["protocol_sha256"][:16] + "...")
    for a in ("E", "E3b", "R4", "F16"):
        al = J(f"configs/protocol_v5_addendum_{a}_lock.json")
        check(f"addendum {a}", sha(ROOT / al["addendum"]) == al["addendum_sha256"] and al["base_protocol_sha256"] == lock["protocol_sha256"],
              f"locked {al['locked_at_utc']}")
    for rel, digest in proto["code"].items():
        check(f"locked code {rel}", sha(ROOT / rel) == digest)
    check("image manifest", sha(ROOT / "reports/analysis/v5_image_manifest.json") == lock["image_manifest_sha256"])
    check("V3 thresholds carry the locked protocol hash", J("reports/analysis/v3_thresholds.json")["protocol_sha256"] == lock["protocol_sha256"])
    check("V4 summary carries the locked protocol hash", J("reports/analysis/v4_summary.json")["protocol_sha256"] == lock["protocol_sha256"])

    if args.wam:
        try:
            head = subprocess.run(["git", "-C", str(args.wam), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            head = "unknown"
        check("WAM commit", head == lock["wam_commit"], head)
        for rel, digest in proto["wam"]["files_sha256"].items():
            p = args.wam / rel
            check(f"WAM {rel}", p.is_file() and sha(p) == digest, "" if p.is_file() else "missing")

    if args.coco:
        files = sorted(args.coco.glob("*.jpg"))
        check("COCO val2017 has 5,000 images", len(files) == 5000, str(len(files)))
        for split, v in proto["images"]["splits"].items():
            a, b = v["index"]
            blob = "".join(f"{f.name}:{sha(f)}\n" for f in files[a:b + 1]).encode()
            check(f"image list {split} ({a}-{b})", hashlib.sha256(blob).hexdigest() == lock["image_lists_sha256"][split])

    print(f"{sum(checks)}/{len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()

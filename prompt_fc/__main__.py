"""Generate Prompt FC.csv from tactical tracking and automatic restart tags."""

import argparse
import hashlib
from pathlib import Path
import platform

import numpy as np

from games import GAME, GAMES, half_video_blocks, require
from starter.load_smart_tagging import load
from .diagnostics import describe, write_json
from .model import Rules, infer, restart_cues
from .tracking import load_tracking

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_submission(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="bundle root (otherwise MTAH_DATA_DIR or data/)")
    parser.add_argument("--out", default=str(ROOT / "evaluation/results/Prompt FC.csv"))
    parser.add_argument("--diagnostics", default=str(
        ROOT / "evaluation/reports/prompt_fc/generation.json"))
    args = parser.parse_args()
    meta, raw, tags = require(["atd_meta", "atd_raw", "smart_tagging"],
                              data_dir=args.data_dir, source="tactical")
    game = GAMES[GAME]
    blocks = half_video_blocks(GAME)
    rules = Rules()
    print("Reading tactical tracking and automatic tags ...", flush=True)
    names, evidence = load_tracking(meta, raw, game["n_video_frames"],
                                     game["pitch"], game["fps"])
    cues = restart_cues(load(data_dir=args.data_dir), names)
    labels, decisions = infer(evidence, cues, blocks, game["pitch"], game["fps"], rules)
    mask = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    if np.any(labels[~mask] != "X") or not np.isin(labels, list("HADX")).all():
        raise ValueError("Invalid output labels / half bounds")
    write_submission(args.out, labels)
    summary = describe(labels, blocks, game["fps"])
    report = {
        "algorithm": "prompt_fc_v1", "game": GAME, "fps": game["fps"],
        "half_blocks_inclusive": blocks, "rules": rules.as_dict(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "input_sha256": {Path(p).name: sha256(p) for p in (meta, raw, tags)},
        "submission_sha256": sha256(args.out),
        "tracking": {"accepted_ball_frames_graded": int(np.sum(
            mask & np.isfinite(evidence.ball).all(axis=1))), "restart_cues": len(cues)},
        "summary": summary, **decisions,
    }
    write_json(args.diagnostics, report)
    print("Wrote {}".format(args.out))
    print("SHA-256 {}".format(report["submission_sha256"]))
    print("Graded label counts: {}".format(summary["label_counts"]))
    print("{} possessions; {} stoppages; {} brief team flips".format(
        summary["possessions"]["count"], summary["stoppages"]["count"],
        len(summary["brief_team_flips_under_2s"])))
    print("Diagnostics: {}".format(args.diagnostics))


if __name__ == "__main__":
    main()

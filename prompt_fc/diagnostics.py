"""Label-only spell diagnostics, independent of any ground truth."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from games import GAME, GAMES, half_video_blocks


def runs(labels, first, last):
    start = first
    for f in range(first + 1, last + 2):
        if f == last + 1 or labels[f] != labels[start]:
            yield str(labels[start]), start, f
            start = f


def describe(labels, blocks, fps):
    spans = [span for _, first, last in blocks for span in runs(labels, first, last)]
    graded = np.concatenate([labels[first:last + 1] for _, first, last in blocks])
    result = {"graded_frames": len(graded),
              "label_counts": {c: int(np.sum(graded == c)) for c in "HADX"}}
    for name, classes in (("possessions", "HA"), ("stoppages", "D")):
        lengths = [(end - start) / fps for label, start, end in spans if label in classes]
        result[name] = {
            "count": len(lengths),
            "duration_s": dict(zip(("min", "p10", "median", "p90", "max"),
                                   np.percentile(lengths, [0, 10, 50, 90, 100]).tolist()))
                          if lengths else {},
            "mean_s": float(np.mean(lengths)) if lengths else 0.0,
            "under_1s": sum(x < 1 for x in lengths),
            "under_2s": sum(x < 2 for x in lengths),
        }
    flips = []
    for half, first, last in blocks:
        seq = list(runs(labels, first, last))
        for prev, middle, after in zip(seq, seq[1:], seq[2:]):
            if (prev[0] in "HA" and middle[0] in "HA" and after[0] == prev[0]
                    and middle[2] - middle[1] < 2 * fps):
                flips.append({"half": half, "start": middle[1], "end": middle[2],
                              "labels": prev[0] + middle[0] + after[0]})
    result["brief_team_flips_under_2s"] = flips
    return result


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission")
    parser.add_argument("--out", default="evaluation/reports/prompt_fc/label_diagnostics.json")
    args = parser.parse_args()
    with open(args.submission, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (len(rows) != GAMES[GAME]["n_video_frames"]
            or any(row != {"frame": str(i), "label": row.get("label")}
                   or row.get("label") not in set("HADX") for i, row in enumerate(rows))):
        raise ValueError("Run evaluation/validate_submission.py on this file")
    labels = np.array([row["label"] for row in rows])
    report = describe(labels, half_video_blocks(GAME), GAMES[GAME]["fps"])
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Read tactical ATD into compact, metre-based evidence arrays."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Evidence:
    ball: np.ndarray             # (frames, 2), metres; missing/rejected = NaN
    distance: np.ndarray         # (frames, 2), nearest home / away, metres
    player_speed: np.ndarray     # frame median of plausible player speeds, m/s
    ball_speed: np.ndarray       # displacement over 0.2 s, m/s


def metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2 or len({t.get("id") for t in teams}) != 2:
        raise ValueError("ATD metadata must declare two distinct teams")
    ids = {team.get("id"): i for i, team in enumerate(teams)}
    players = list(root.iter("Player"))
    if not players or any(p.get("teamId") not in ids for p in players):
        raise ValueError("Every ATD track slot must have a declared team")
    return (tuple(team.findtext("Name") for team in teams),
            np.array([ids[p.get("teamId")] for p in players]))


def clean_ball(ball, pitch, fps):
    """Reject impossible positions and isolated teleports, never interpolate gaps.

    A 65 m/s ceiling plus 1.5 m measurement tolerance is deliberately generous
    for a kicked ball. Require a plausible neighbour within 0.5 s; a point on
    its own cannot establish control or a boundary crossing.
    """
    out = ball.copy()
    plausible = (np.isfinite(out).all(axis=1)
                 & (out >= -12).all(axis=1)
                 & (out <= np.asarray(pitch) + 12).all(axis=1))
    frames = np.flatnonzero(plausible)
    supported = np.zeros(len(out), dtype=bool)
    if len(frames) > 1:
        dt = np.diff(frames) / fps
        step = np.linalg.norm(np.diff(out[frames], axis=0), axis=1)
        linked = (dt <= 0.5) & (step <= 65 * dt + 1.5)
        supported[frames[:-1][linked]] = True
        supported[frames[1:][linked]] = True
    out[~(plausible & supported)] = np.nan
    return out


def load_tracking(meta_path, raw_path, n_frames, pitch, fps):
    names, teams = metadata(meta_path)
    ball = np.full((n_frames, 2), np.nan)
    distance = np.full((n_frames, 2), np.inf)
    player_speed = np.full(n_frames, np.nan)
    scale = np.asarray(pitch)
    count = 0
    with open(raw_path, encoding="utf-8") as handle:
        for frame, line in enumerate(handle):
            parts = line.strip().split(":")
            if len(parts) != 3 or int(parts[0]) != frame + 1 or frame >= n_frames:
                raise ValueError("ATD must contain ordered one-based IDs 1..n; bad row {}"
                                 .format(frame + 1))
            players = np.fromstring(parts[1].replace(";", ","), sep=",")
            b = np.fromstring(parts[2], sep=",")
            if len(players) != len(teams) * 3 or len(b) != 3:
                raise ValueError("ATD slot count mismatch at row {}".format(frame + 1))
            players = players.reshape(-1, 3)
            xy = players[:, :2] * scale
            valid = (np.isfinite(xy).all(axis=1) & (xy >= -5).all(axis=1)
                     & (xy <= scale + 5).all(axis=1))
            speeds = players[:, 2]
            speed_ok = valid & np.isfinite(speeds) & (speeds >= 0) & (speeds <= 15)
            if np.sum(speed_ok) >= 6:
                player_speed[frame] = np.median(speeds[speed_ok])
            ball[frame] = b[:2] * scale
            if np.isfinite(ball[frame]).all():
                d = np.linalg.norm(xy - ball[frame], axis=1)
                for team in (0, 1):
                    candidates = d[valid & (teams == team)]
                    if len(candidates):
                        distance[frame, team] = np.min(candidates)
            count += 1
    if count != n_frames:
        raise ValueError("Expected {} ATD rows, found {}".format(n_frames, count))
    ball = clean_ball(ball, pitch, fps)
    distance[~np.isfinite(ball).all(axis=1)] = np.inf
    lag = max(1, round(0.2 * fps))
    ball_speed = np.full(n_frames, np.nan)
    ball_speed[lag:] = np.linalg.norm(ball[lag:] - ball[:-lag], axis=1) * fps / lag
    return names, Evidence(ball, distance, player_speed, ball_speed)

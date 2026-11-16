"""Independent live/dead inference followed by hysteretic team control.

All constants describe distances or durations, chosen before public scoring.
Tag intervals are approximate restart search windows, never possession labels.
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Rules:
    control_radius_m: float = 2.5
    advantage_m: float = 1.0
    control_speed_mps: float = 15.0
    control_s: float = 0.6
    evidence_gap_s: float = 0.4
    boundary_margin_m: float = 0.4
    outside_s: float = 0.24
    stationary_s: float = 5.0
    stationary_radius_m: float = 0.8
    slow_players_mps: float = 1.5
    restart_inside_s: float = 0.4
    restart_motion_m: float = 2.0
    tag_slack_s: float = 3.0

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Restart:
    start: int
    end: int                   # exclusive
    team: int                  # 0=H, 1=A, -1=unknown (including N/A)
    kind: str


def restart_cues(events, names):
    teams = {name.strip().casefold(): i for i, name in enumerate(names)}
    cues = []
    for event in events:
        if (event["code"] or "").strip().casefold() != "set pieces":
            continue
        tags = event["tags"]
        if event["end_frame"] <= event["start_frame"]:
            continue
        cues.append(Restart(event["start_frame"], event["end_frame"],
                            teams.get((tags.get("Team") or "").strip().casefold(), -1),
                            (tags.get("Type") or "").strip().casefold()))
    return sorted(cues, key=lambda c: (c.start, c.end, c.kind, c.team))


def control_evidence(evidence, pitch, rules):
    d = evidence.distance
    nearest = np.argmin(d, axis=1)
    own = d[np.arange(len(d)), nearest]
    other = d[np.arange(len(d)), 1 - nearest]
    inside = (np.isfinite(evidence.ball).all(axis=1)
              & (evidence.ball >= 0).all(axis=1)
              & (evidence.ball <= np.asarray(pitch)).all(axis=1))
    # An unavailable speed is uncertainty, not proof of a fast pass.
    slow = ~np.isfinite(evidence.ball_speed) | (evidence.ball_speed <= rules.control_speed_mps)
    margin = np.zeros(len(d))
    both = np.isfinite(own) & np.isfinite(other)
    margin[both] = other[both] - own[both]
    clear = inside & slow & both & (own <= rules.control_radius_m) & (margin >= rules.advantage_m)
    return np.where(clear, nearest, -1), inside


def dead_ball(evidence, inside, cues, blocks, pitch, fps, rules):
    """Confirm out-of-play / stationary setups and release on credible motion.

    Missing tracking alone cannot start or end D. Stationary inference requires
    a ball near a player AND low collective player motion; this also handles
    some untagged stoppages. Kickoff setup is restricted to the half's start.
    """
    n = len(inside)
    dead = np.zeros(n, dtype=bool)
    reasons = []
    ball = evidence.ball
    seen = np.isfinite(ball).all(axis=1)
    outside = seen & ((ball < -rules.boundary_margin_m).any(axis=1)
                      | (ball > np.asarray(pitch) + rules.boundary_margin_m).any(axis=1))
    slack = round(rules.tag_slack_s * fps)
    need_out = max(2, round(rules.outside_s * fps))
    need_in = max(2, round(rules.restart_inside_s * fps))
    max_gap = round(rules.evidence_gap_s * fps)
    stationary_frames = round(rules.stationary_s * fps)

    for half, first, last in blocks:
        half_cues = [c for c in cues if c.end > first and c.start <= last]
        # A start-of-half tag can be minutes into the graded block. Do not use
        # games' video kickoff anchors as hidden labels or transfer half 1 state.
        kickoff = next((c for c in half_cues if c.kind == "kick off (start)"), None)
        state = kickoff is not None
        start_dead = first if state else None
        reason = "kickoff_setup" if state else None
        out_start = out_last = None
        out_count = 0
        anchor = None
        still_start = still_last = None
        still_count = slow_count = 0
        in_start = in_last = None
        in_count = 0
        motion_anchor = None
        motion_start = None

        for f in range(first, last + 1):
            if state:
                dead[f] = True
                # A kickoff cue gates release from the pre-half setup, but
                # tracking, not the tag centre, chooses the exact restart.
                eligible = reason != "kickoff_setup" or f >= kickoff.start - slack
                if eligible and inside[f]:
                    if in_last is None or f - in_last > max_gap:
                        in_start, in_count, motion_anchor = f, 0, ball[f].copy()
                        motion_start = None
                    in_last = f
                    in_count += 1
                    displacement = np.linalg.norm(ball[f] - motion_anchor)
                    if motion_start is None and displacement >= 0.5:
                        motion_start = f
                    moved = displacement >= rules.restart_motion_m
                    nearby = np.min(evidence.distance[f]) <= 8.0
                    if in_count >= need_in and moved and nearby:
                        # A ball can sit inside the field for a long time before
                        # a free kick. Backdate only to observed motion onset.
                        restart = max(in_start, motion_start - round(0.2 * fps))
                        dead[restart:f + 1] = False
                        reasons.append({"half": half, "start": start_dead,
                                        "end": restart, "reason": reason})
                        state = False
                        anchor = None
                        still_start = still_last = None
                        still_count = slow_count = 0
                        out_start = out_last = None
                        out_count = 0
                        in_last = None
                elif seen[f]:
                    in_last = None
                continue

            if outside[f]:
                if out_last is None or f - out_last > max_gap:
                    out_start, out_count = f, 0
                out_last = f
                out_count += 1
            elif seen[f]:
                out_start = out_last = None
                out_count = 0
            if out_count >= need_out:
                state, start_dead, reason = True, out_start, "boundary"
            else:
                if inside[f]:
                    if (anchor is None or still_last is None or f - still_last > max_gap
                            or np.linalg.norm(ball[f] - anchor) > rules.stationary_radius_m):
                        anchor = ball[f].copy()
                        still_start = f
                        still_count = slow_count = 0
                    still_last = f
                    still_count += 1
                    slow_count += int(np.isfinite(evidence.player_speed[f])
                                      and evidence.player_speed[f] < rules.slow_players_mps)
                    if (f - still_start >= stationary_frames
                            and still_count >= 0.75 * (f - still_start + 1)
                            and slow_count >= 0.75 * still_count
                            and np.min(evidence.distance[f]) <= 3.0):
                        state, start_dead, reason = True, still_start, "stationary_setup"
                elif seen[f]:
                    anchor = None
                    still_last = None
            if state:
                dead[start_dead:f + 1] = True
                in_start = in_last = None
                in_count = 0
                motion_anchor = None
                motion_start = None
        if state:
            reasons.append({"half": half, "start": start_dead,
                            "end": last + 1, "reason": reason})
    return dead, reasons


def infer(evidence, cues, blocks, pitch, fps, rules=Rules()):
    control, inside = control_evidence(evidence, pitch, rules)
    dead, stoppages = dead_ball(evidence, inside, cues, blocks, pitch, fps, rules)
    labels = np.full(len(inside), "X", dtype="<U1")
    need = max(2, round(rules.control_s * fps))
    max_gap = round(rules.evidence_gap_s * fps)
    slack = round(rules.tag_slack_s * fps)
    switches = []
    for half, first, last in blocks:
        current = pending = -1
        count = 0
        pending_start = pending_last = None
        live_start = first
        was_dead = False
        for f in range(first, last + 1):
            if dead[f]:
                labels[f] = "D"
                current = pending = -1
                count = 0
                pending_last = None
                was_dead = True
                continue
            if was_dead or f == first:
                live_start = f
                # A known tagged side is a prior only at a tracking-established
                # restart. Conflicting or missing Team tags abstain.
                nearby = {c.team for c in cues if c.team >= 0
                          and c.end > first and c.start <= last
                          and c.start - slack <= f < c.end + slack}
                current = next(iter(nearby)) if len(nearby) == 1 else -1
                was_dead = False
            team = int(control[f])
            if ((team >= 0 and team == current)
                    or (pending_last is not None and (f - pending_last > max_gap
                        or f - pending_start > 2 * need))):
                pending, count, pending_last = -1, 0, None
            if team >= 0 and team != current:
                if team != pending:
                    pending, count, pending_start = team, 0, f
                count += 1
                pending_last = f
                if count >= need:
                    start = live_start if current < 0 else pending_start
                    labels[start:f + 1] = "HA"[team]
                    switches.append({"half": half, "frame": start, "confirmed": f,
                                     "from": "HA"[current] if current >= 0 else "X",
                                     "to": "HA"[team]})
                    current = team
                    pending, count, pending_last = -1, 0, None
            labels[f] = "HA"[current] if current >= 0 else "X"
    return labels, {"stoppages": stoppages, "control_confirmations": switches}

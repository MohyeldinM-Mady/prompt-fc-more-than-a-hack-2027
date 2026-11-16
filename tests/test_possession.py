"""Synthetic football scenarios; no public or private match labels are read."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from games import GAME, half_video_blocks
from prompt_fc.__main__ import write_submission
from prompt_fc.model import Rules, Restart, infer, restart_cues
from prompt_fc.tracking import Evidence, clean_ball, load_tracking
from starter.load_smart_tagging import load

PITCH = (105.3, 68.0)
FPS = 25


def evidence(n=250):
    return Evidence(np.column_stack((30 + np.arange(n) * .1, np.full(n, 30.))),
                    np.tile([1., 6.], (n, 1)), np.full(n, 2.), np.full(n, 2.5))


def predict(e, cues=(), blocks=None):
    return infer(e, cues, blocks or [(1, 0, len(e.ball) - 1)], PITCH, FPS)


class PossessionTests(unittest.TestCase):
    def test_missing_ball_carries_team_without_dead_ball(self):
        e = evidence()
        e.ball[50:150] = np.nan
        e.distance[50:150] = np.inf
        labels, _ = predict(e)
        self.assertTrue(np.all(labels == "H"))

    def test_contested_and_brief_opponent_touches_keep_control(self):
        e = evidence()
        e.distance[50:100] = [1., 1.2]
        e.distance[110:120] = [6., 1.]
        labels, _ = predict(e)
        self.assertTrue(np.all(labels == "H"))

    def test_sustained_control_switch_is_backdated(self):
        e = evidence()
        e.distance[80:] = [6., 1.]
        labels, decisions = predict(e)
        self.assertTrue(np.all(labels[:80] == "H"))
        self.assertTrue(np.all(labels[80:] == "A"))
        self.assertEqual(decisions["control_confirmations"][-1]["confirmed"], 94)

    def test_missing_opposing_players_cannot_confirm_a_switch(self):
        e = evidence()
        e.distance[80:] = [np.inf, 1.]
        labels, _ = predict(e)
        self.assertTrue(np.all(labels == "H"))

    def test_fast_pass_does_not_grant_opponent_control(self):
        e = evidence()
        e.distance[80:120] = [6., 1.]
        e.ball_speed[80:120] = 25
        labels, _ = predict(e)
        self.assertTrue(np.all(labels == "H"))

    def test_sparse_evidence_does_not_accumulate_forever(self):
        e = evidence(400)
        e.distance[50:] = [5., 5.]
        e.distance[50::9] = [6., 1.]
        labels, _ = predict(e)
        self.assertTrue(np.all(labels == "H"))

    def test_boundary_stoppage_and_untagged_restart(self):
        e = evidence()
        e.ball[50:100, 1] = 69
        e.ball[100:, 1] = 60 - np.arange(150) * .1
        labels, decisions = predict(e)
        self.assertEqual(labels[49], "H")
        self.assertTrue(np.all(labels[50:100] == "D"))
        self.assertEqual(labels[125], "H")
        self.assertEqual(decisions["stoppages"][0]["reason"], "boundary")

    def test_one_noisy_outside_frame_is_not_a_stoppage(self):
        e = evidence()
        e.ball[50, 1] = 69
        labels, _ = predict(e)
        self.assertTrue(np.all(labels == "H"))

    def test_stationary_ball_with_slow_players_detects_untagged_stop(self):
        e = evidence(350)
        e.ball[50:240] = [40, 30]
        e.player_speed[50:240] = .5
        e.ball[240:] = np.column_stack((40 + np.arange(110) * .2, np.full(110, 30.)))
        labels, decisions = predict(e)
        self.assertTrue(np.all(labels[50:230] == "D"))
        self.assertEqual(labels[260], "H")
        self.assertEqual(decisions["stoppages"][0]["reason"], "stationary_setup")

    def test_stationary_tracking_with_running_players_does_not_imply_dead(self):
        e = evidence()
        e.ball[:] = [40, 30]
        labels, _ = predict(e)
        self.assertTrue(np.all(labels == "H"))

    def test_each_half_resets_and_delayed_kickoff_remains_dead(self):
        e = evidence(300)
        e.distance[150:] = [6., 1.]
        e.ball[150:230] = [50, 34]
        e.ball[230:] = np.column_stack((50 + np.arange(70) * .2, np.full(70, 34.)))
        labels, _ = predict(e, [Restart(220, 250, 1, "kick off (start)")],
                            [(1, 10, 99), (2, 150, 289)])
        self.assertTrue(np.all(labels[10:100] == "H"))
        self.assertTrue(np.all(labels[150:220] == "D"))
        self.assertTrue(np.all(labels[250:290] == "A"))
        self.assertTrue(np.all(labels[100:150] == "X"))
        self.assertEqual(labels[9], "X")
        self.assertEqual(labels[290], "X")

    def test_restart_tag_is_a_prior_not_a_live_possession_override(self):
        e = evidence()
        labels, _ = predict(e, [Restart(70, 180, 1, "corner kick")])
        self.assertTrue(np.all(labels == "H"))

    def test_kickoff_tag_can_start_before_inclusive_graded_bound(self):
        e = evidence()
        e.ball[:50] = [40, 30]
        e.ball[50:] = np.column_stack((40 + np.arange(200) * .2, np.full(200, 30.)))
        labels, _ = predict(e, [Restart(0, 90, 0, "kick off (start)")],
                            blocks=[(1, 40, 249)])
        self.assertEqual(labels[39], "X")
        self.assertTrue(np.all(labels[40:48] == "D"))
        self.assertEqual(labels[80], "H")

    def test_unknown_initial_control_abstains_and_halves_do_not_leak(self):
        e = evidence(150)
        e.distance[100:] = [5., 5.]
        labels, _ = predict(e, blocks=[(1, 0, 49), (2, 100, 149)])
        self.assertTrue(np.all(labels[:50] == "H"))
        self.assertTrue(np.all(labels[100:] == "X"))

    def test_ball_cleaning_rejects_teleport_and_retains_missing_gap(self):
        ball = evidence(100).ball
        ball[30] = [100, 65]
        ball[50:60] = np.nan
        cleaned = clean_ball(ball, PITCH, FPS)
        self.assertTrue(np.isnan(cleaned[30]).all())
        self.assertTrue(np.isnan(cleaned[50:60]).all())
        self.assertTrue(np.isfinite(cleaned[29]).all())

    def test_tag_names_unknown_teams_and_seconds_rounding(self):
        xml = ('<root><instance><start>1.06</start><end>2.02</end><code>SET PIECES</code>'
               '<label><group>Team</group><text>N/A</text></label></instance></root>')
        with tempfile.TemporaryDirectory() as folder:
            tactical = Path(folder) / "tactical"
            tactical.mkdir()
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(xml)
            events = load(data_dir=folder)
        cue = restart_cues(events, ("Barcelona", "Atletic Club"))[0]
        self.assertEqual((cue.start, cue.end, cue.team), (round(1.06 * 25), round(2.02 * 25), -1))

    def test_raw_alignment_and_strict_malformed_input(self):
        meta = ('<root><Team id="h"><Name>H</Name></Team><Team id="a"><Name>A</Name>'
                '</Team><Player teamId="h"/><Player teamId="a"/></root>')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "meta.xml").write_text(meta)
            (path / "raw.txt").write_text(
                '1:0.5,0.5,0;0.8,0.5,0:0.51,0.5,0\n'
                '2:0.5,0.5,0;0.8,0.5,0:0.52,0.5,0\n')
            names, e = load_tracking(path / "meta.xml", path / "raw.txt", 2, PITCH, FPS)
            self.assertAlmostEqual(e.ball[0, 0], .51 * PITCH[0])
            self.assertEqual(names, ("H", "A"))
            (path / "raw.txt").write_text('2:0.5,0.5,0;0.8,0.5,0:0.51,0.5,0\n')
            with self.assertRaisesRegex(ValueError, "one-based"):
                load_tracking(path / "meta.xml", path / "raw.txt", 2, PITCH, FPS)

    def test_full_contract_bounds_and_canonical_csv_bytes(self):
        self.assertEqual(half_video_blocks(GAME), [(1, 11185, 90651), (2, 110808, 190197)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.csv"
            write_submission(path, ["X", "H", "A", "D"])
            self.assertEqual(path.read_bytes(), b"frame,label\n0,X\n1,H\n2,A\n3,D\n")


if __name__ == "__main__":
    unittest.main()

# Prompt FC — Phase 1 technical note

## Approach and inputs

Prompt FC v1 is a deterministic, untrained state machine. It separates
live/dead inference from team control so that losing sight of the ball does
not automatically create a stoppage. A second state machine preserves the
last controlling team through loose balls, tracking gaps and contested touches.
Sustained opposing control establishes a switch. Offline confirmation allows
backdating to the beginning of the evidence rather than adding confirmation
delay to every possession boundary.

Inputs are the **tactical** ATD metadata XML, raw ATD TXT and automatic Smart
Tagging XML, located through `games.require()` and the existing tag loader.
No video is available in the local bundle. Broadcast tracking is unused because
it has a different timeline; homographies are unused because ATD already supplies
pitch coordinates. No detector, pretrained model, external dataset, human
annotation or training step is used. The public five-minute labels are used
only by the official scorer after fixing the algorithm, for a pipeline check.
They are never loaded by the generator or synthetic scenario tests.

Metadata assigns track slots to the two teams (first team = H, second = A).
Slots are not persistent player identities. Coordinates are scaled from 0–1 to
105.3 × 68 metres. `video_frame = raw_frame_id - 1`; ordered IDs and the full
199,319-record input are checked strictly. Tags use the existing conversion
`round(seconds * 25)`, with an exclusive end. `games.half_video_blocks()` supplies
inclusive bounds, 11185–90651 and 110808–190197. State is reset in each half and
all other frames are X. No video kickoff anchor is read by the inference code.

## Possession rules

The nearest player on **each** team is computed for every accepted ball
observation. Control evidence requires:

- the ball inside the pitch;
- a player within 2.5 m, at least 1 m closer than the other team's closest player;
- both teams represented by valid tracks;
- ball displacement speed no greater than 15 m/s when measurable.

A switch needs 15 qualifying observations (0.6 s at 25 fps), no evidence gap
longer than 0.4 s, and an evidence span no longer than 1.2 s. Control evidence
for the current team cancels an opposing pending switch. Uncertain frames do
not vote for the opponent or D. On confirmation, the switch is backdated to
the first supporting frame. A new live spell starts with a known restart prior
or X until control is confirmed, then its unresolved prefix is filled with that
team. Confirmed live possession is carried through tracking loss, even long
losses, until contrary control or stoppage evidence arrives.

The metre and time thresholds express a playable control neighbourhood,
separation from an opponent and persistence beyond an isolated touch. They
are fixed engineering assumptions, not trained estimates or a public-score
parameter search. Parameters are declared in `prompt_fc/model.py`; every run
records them in its ignored diagnostics.

## Dead ball and restarts

The observation filter rejects nonfinite ball positions, positions more than
12 m beyond the pitch, and isolated jumps with no plausible neighbouring
observation within 0.5 s. A generous 65 m/s travel ceiling with 1.5 m positional
tolerance rejects isolated teleports without trying to reconstruct the ball.
Player positions more than 5 m outside the pitch are excluded. There is no ball
interpolation. Missing player or ball speeds remain uncertainty.

Two tracking conditions can start D:

1. The ball is more than 0.4 m outside a boundary on six supported observations
   (0.24 s), with no gap longer than 0.4 s and no intervening observed return.
   D is backdated to the first outside observation.
2. The ball stays within 0.8 m of a fixed anchor for at least 5 s, at least 75%
   of the frames have observations, and at least 75% of those have median player
   speed below 1.5 m/s. At confirmation there must be a player within 3 m.
   Median speed requires at least six plausible player speeds (0–15 m/s).
   D is backdated to the stationary setup. This can detect an untagged stoppage.

An isolated outside point, missing tracking, a stationary ball with running
players, or a set-piece tag alone does not establish a normal stoppage.

The 59 set-piece tags are approximate **restart windows**, not intervals to
copy as D or team possession. A half's overlapping `Kick Off (Start)` cue
initializes D before its restart, including the lengthy pre-kickoff portion
of the second graded block. Tag starts may precede the first graded frame.
Kickoff release becomes eligible 3 s before the tag starts. No half's state
is inherited from the previous half.

To release D, require at least ten inside observations, no gap longer than
0.4 s, displacement of at least 2 m from the inside anchor and a player within
8 m at confirmation. Release is backdated to the first observed displacement
of 0.5 m, minus at most 0.2 s. This avoids assigning a long stationary setup
to live play. Otherwise D persists through uncertainty. This release rule
works without a restart tag.

At a tracking-established restart, a unique known tagged team within the
tag interval expanded by 3 s supplies the initial possession prior. Club names
are mapped through ATD metadata, case-insensitively. `N/A`, unknown teams and
conflicting priors abstain. Sustained tracking control can override a prior.
Tags encountered during an established live possession cannot force a switch.

## Verification and observed limitations

The frozen candidate's official **public-only** check gives:

| Submission | spellF1 (ranking) | macroF1 (first tie-break) | Public possessions |
|---|---:|---:|---:|
| Prompt FC | 0.3647 | 0.5576 | 17 |
| Organizer baseline_smooth50 | 0.3256 | 0.6207 | 26 |

This is a five-minute pipeline check, not evidence of full-match superiority.
No settings or candidate were selected by these scores. Private labels were
not accessed. The organizer's private baseline score is not comparable with
our public score.

Label-only full-match diagnostics, restricted to the 158,857 graded frames:

| Diagnostic | Prompt FC | baseline_smooth50 |
|---|---:|---:|
| Possession spells | 298 | 594 |
| Mean / median possession duration (s) | 18.80 / 10.86 | 7.24 / 4.60 |
| Possession duration p10 / p90 (s) | 1.84 / 40.84 | 0.44 / 18.05 |
| Possessions under 1 s | 5 | 124 |
| H–A–H or A–H–A flips with middle spell under 2 s | 25 | 105 |
| Stoppages | 52 | 133 |
| Median / longest stoppage (s) | 5.18 / 138.92 | 7.56 / 217.48 |

The candidate infers 46 boundary stoppages, four stationary setups and two
kickoff setups. Counts are predictions, not corrected match statistics.
Its graded labels are H=71,115, A=68,917, D=18,643, X=182; the additional
40,462 X frames are outside the graded blocks. Accepted ball observations
cover 93,942 graded frames. Detailed spell distributions, brief flips,
stoppage boundaries and confirmation decisions are retained locally in
`evaluation/reports/prompt_fc/generation.json`.

Dead-ball recall is the main limitation: on the public check it is 6.81%,
despite 97.09% D precision. In-field whistles, unobserved boundary exits and
stoppages with missing ball tracking are often missed. Conversely, a restart
without observable motion may leave D active too long. Live possession can
also persist too long through missing tracks. Stationary tracking errors can
look like setups, and goalkeeper holds can be confused with stoppages.
No ball height, referee signal or video evidence disambiguates those cases.
Automatic tags may be mistimed, omitted or wrong, and throw-ins lack a team
prior. Track duplication and team errors can still create false control.
The next improvements would add algorithmic video/restart evidence and
evaluate on independently labelled **other games**, not tune these thresholds
on this match's public labels.

## Reproduction

Tested on Windows with **CPython 3.14.5**, numpy 2.5.4 and pandas 3.1.1. The full
candidate/scorer runtime dependencies are pinned in `requirements-prompt-fc.txt`.
The candidate imports only numpy beyond the standard library; notebook, kloppy
and video packages are unnecessary. A fresh environment installation and
regeneration were checked separately from the existing development environment.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-prompt-fc.txt
# Optional: bundle parent directory containing tactical/:
# $env:MTAH_DATA_DIR = "D:\match-bundle"
.\.venv\Scripts\python.exe -m prompt_fc
.\.venv\Scripts\python.exe evaluation/validate_submission.py "evaluation/results/Prompt FC.csv"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe evaluation/run_evaluation.py --no-detail --out evaluation/reports/prompt_fc/public
```

The exact default output is `evaluation/results/Prompt FC.csv`, UTF-8 with LF
newlines and header `frame,label`, frames 0–199318 in order. Its SHA-256 is:

```text
665abff4327e11323de9b7423f920581465ef4ad81019b8a49535e20dad2fed3
```

There is no random state, timestamp, cache or ground-truth access in generation.
Input file hashes and runtime versions are recorded in the ignored JSON report.
To independently rebuild and compare exact bytes:

```powershell
.\.venv\Scripts\python.exe -m prompt_fc --out "evaluation/reports/prompt_fc/repeat/Prompt FC.csv" --diagnostics evaluation/reports/prompt_fc/repeat/generation.json
.\.venv\Scripts\python.exe -c "from pathlib import Path; a=Path('evaluation/results/Prompt FC.csv').read_bytes(); b=Path('evaluation/reports/prompt_fc/repeat/Prompt FC.csv').read_bytes(); assert a == b; print('Byte-identical')"
```

The unchanged organizer fallback can be regenerated separately:

```powershell
.\.venv\Scripts\python.exe starter/baseline.py --window 50 --out evaluation/reports/fallback
.\.venv\Scripts\python.exe evaluation/validate_submission.py evaluation/reports/fallback/baseline_smooth50.csv
.\.venv\Scripts\python.exe -m prompt_fc.diagnostics evaluation/reports/fallback/baseline_smooth50.csv --out evaluation/reports/prompt_fc/baseline_diagnostics.json
```

Match assets remain under ignored `data/` (or outside the repository). Candidate
CSVs remain under ignored `evaluation/results/` or `evaluation/reports/`. Detailed
reports remain under ignored `evaluation/reports/`; they are not public exports.
Notebook outputs must be cleared before public commits to exclude embedded
match imagery. Existing local notebook edits are separate from this pipeline.
Publish the source, pinned requirements and this note in the team repository;
upload the ignored CSV and this Markdown note through the organizer's form.

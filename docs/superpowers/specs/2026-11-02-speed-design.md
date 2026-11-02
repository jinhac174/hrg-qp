# Speed: acceleration-bounded planning, inertia feed-forward, a continuous gripper

Status: asked for by the user on 2026-11-02 ("very reactive and very fast ... not a small velocity
upgrade ... what controls the speed ... the gripper twitch ... no band-aid patches"). Built and
validated in the real-time simulation the same day. docs/method.md ("Speed") has the user-facing
account; this is the design record.

## Why

The method is first order: the fields create a velocity, the QP tracks it under constraints, the
low level realises it. Three things kept it slow and would have made a plain speed-up unsafe:

1. **The QP has no notion of acceleration.** Its velocity may jump between plans. The distance
   rows, linearised over `H`, assume the velocity is realised within `H`; a torque-controlled arm
   realises a jump with a lag, so at speed the margins are overshot by `v × lag`.
2. **The paper's low level is a Kd velocity loop** (Kp = 0 on the one-interval integration). Its
   time constant `M_ii / Kd_i` is 30–60 ms on the FR3's joints 1–4 at gains that are quiet on the
   robot (measured 25–31 ms lag). That lag set `H = 0.10 s`, and the margin servo `(d − ε)/H` then
   governs the whole final approach.
3. **The gripper stream raced.** Width set-points went out at 50 Hz at the gripper's full speed:
   the fingers reached each in ~11 ms and waited (a third of the stream stopped, measured). On the
   real 2F-85 that is a twitch.

## What was built

* **Acceleration bounds in the QP** (`planner._bounds`): `|z − z_prev| <= a Δt_plan`, `a` the
  FR3's joint acceleration limits and 2 m/s² for the width, `z_prev` the previous plan (reset or
  `set_velocity` anchors it at the arm's actual velocity). Every plan (reach, lift, home, hold)
  goes through it; the hold is now the planner's joint field, not a clip in the controller.
* **Distance rows: hard first, slacks on infeasibility** (`qp.solve_tracking_qp`): the hard
  QP as before; if it is infeasible, slacks `s >= 0` with the linear penalty `ρ Σ s` (`ρ = 100`,
  exact above the tracking cost's gradient) plus a small `|s|²` so OSQP converges. Needed because
  hard rows + a hard acceleration box are infeasible whenever a row inside its margin demands more
  separation rate than the arm can reach in one period (the held bowl's table rows at the start of
  the lift: the lift stood still). Solving the slack form always was tried first: slower, and with
  the box's 95 rows OSQP ran out of iterations and the reach stuttered. A failed solve brakes at
  the acceleration limit instead of stopping dead.
* **Low level** (`hrg/lowlevel.py`): the plan's velocity is ramped linearly from the previous
  plan's over the plan period (its acceleration is the QP's), and `M(q) qdd_cmd` is fed forward.
  Lag 2–9 ms on every joint (was 25–31 on joints 2 and 4). The first-order filter is gone.
* **Gripper** (`lowlevel.WidthStream`): set-points a little ahead of the integrated width at the
  speed `|wdot|`; the fingers move continuously at the commanded rate (2 % of the stream stopped).
  The posture field **pre-shapes** to the chosen grasp's opening + 1 cm per side once the grasp is
  chosen, so the width never limits the straight approach. The close runs at the gripper's top
  speed.
* **Fast defaults**: `v_const` 0.8 m/s, `δ_x` 8 cm, alignment 5 /s capped at 2.5 rad/s, posture
  3 /s, lift 0.8 m/s with gain 8 (0.4 s), hold 10 /s at 1 rad/s, home 2 /s, `v_gr` 0.15 m/s.
* **Carried object** (`hrg/carry.py`, env key `m`, `--carry_at`): the object welded to a mocap
  carrier gliding along a C² spline through random waypoints (mid-air included) under a minimum-jerk
  time law, held out at the end, released when both pads touch it. `configs/sim/env_carry.yaml`
  for the longer demo path.
* `scripts/diagnose_tracking.py`: records the bus during one reach and reports the command→motion
  lag per joint, peak speeds / accelerations, and the gripper stream's smoothness.

## Round two: the carried object (same day)

The first videos of a carried object showed the arm shaking and the gripper slow. Measured, not
guessed (`scripts/diagnose_tracking.py --carry_at`, kinematic replays of the recorded carry):

* **The gripper** was slow only in sim: `_GripperSim.command` restarted its set-point ramp at the
  lagging measured width on every streamed set-point, so the servo was 3 mm ahead of the fingers
  per period and they crept at 17 mm/s whatever speed was asked for. Fixed in the sim model (it
  re-targets a running motion like the real 2F-85). Grasp time on the static bowl: 2.7 s → 1.0 s.
* **The shaking** had six causes, removed one by one, each checked by the fraction of plans at the
  acceleration bounds in a replay: the on/off approach corridor (→ a continuous goal on the
  approach segment), a goal behind the hand (→ a depth floor), the 1.5 mm first path segment as the
  direction (→ 2 cm lookahead), the 30 Hz pose hold (→ `PosePredictor`), the object's velocity
  unknown to the rows and fields (→ drift term, target feed-forward), the arm holding still during
  the close while the object moved into the palm (→ the `close` state holds the hand's pose on the
  object with the lift's pose field; a first version kept the reach's fingertip tasks and object
  rows on during the close, which dragged the hand as the pads closed and tipped boxes 60–80°). And the
  demo's own carrier released the object only after its path ended, so a grasp mid-carry had the
  kinematic carrier drag the arm (→ release whenever both pads hold it).
* **Speed** was lowered from 0.8 to 0.5 m/s (δ 10 cm): the eq. (5) decay's own peak deceleration
  `1.5 v² / δ` must fit the arm's acceleration (3.8 m/s² now; 12 before). Static reach 0.65 → 0.9 s
  kinematically; the user accepted the trade.

## The governing relation

`v <= a H`. A distance row allows approach at `(d − ε)/H`, so braking must deliver `v/H`; with
the hand's acceleration `a` (5–8 m/s² at the FR3 joint limits over the arm's lever) and
`H = 0.10 s`, `v_const = 0.8 m/s` is the bound, and `δ_x >= v²/(2a)` ≈ 6 cm keeps the eq. (5)
decay itself feasible. `H` is now the stopping time, not a latency cover. More speed means more
acceleration or a longer horizon (braking earlier).

## Anti-goals

No thresholds or state rules were added. Everything is a constraint the robot has (acceleration), a
penalty formulation that keeps a constrained problem solvable, the dynamics the paper's PD left out
(inertia), or the gripper's own interface (speed per set-point).

## Measured (real-time sim, `docs/results.md`)

Gold bowl, left-bottom start: grasped 2.7 s after `v`, lifted 0.4 s later (was 4.0 s and 1.3 s).
Lag 2–9 ms per joint. Gripper stream 2 % stopped (was 33 %). The evaluation batches of the Table I
set, Fig. 7, Fig. 8 and carried objects are in `docs/results.md`.

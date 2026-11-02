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

## Round three: the first runs on the robot (same day)

The user ran the gold bowl on the FR3 ("I love the speed, but ..."). Two things, measured in the
logs and reproduced in the real-time simulation before anything was changed:

* **The bounce after the lift** ("it lifts up with too much ease, overshoots up a little and comes
  down"). The lift ends within 1 cm of its goal while the hand still rises; the upright turn then
  took the hand's *current* position as its goal, and the hold the joints at its entry: each state
  re-anchored the goal at a point the moving hand had to overshoot and come back to (2 mm in the
  simulation, more on the robot, which follows with more lag). Fix: one goal pose from the grasp on
  (the lift's position for the upright turn, the final pose for the hold, the same pose field).
  The height now rises monotonically and stays.
* **The robot stopping while chasing a bowl moved by hand** (`joint_velocity_violation`, then
  `power_limit_violation`; the hardware log). Two causes: the target feed-forward added the object's
  speed to the fields' (the hand went to 1.1 m/s, the arm at its 2 rad/s box), and the box itself
  was not the robot's. The FR3's joint velocity limits depend on position and fall toward the ends
  of the range (libfranka 0.17.0); in a simulated chase joint 4 reached 135 % of its limit at
  −2.68 rad. And the acceleration limits of 15 / 20 rad/s² on the wrist were not the FR3's (10 on
  every joint). Fix: the QP plans within 0.8 of the FR3's position-dependent limits at 10 rad/s²,
  the fingertips never faster than `v_const` (an object moved faster is caught as it slows), the
  hardware brakes above 0.9 of the same limits. Simulated chase: 82 % at most.
* The hardware process did not notice a reflex (panda-py's torque loop ends in its own thread),
  so every stop needed a restart. It now reports the robot's reasons, saves its last 3 s
  (`outputs/logs/reflex_<time>.npz`) and recovers with `r`. The power limit's definition is not
  published; the next stop's record shows what the arm did.

## Round four: the torque the plans asked for (same day)

The user ran it again with the bus recorded ("Same problem ... Acceleration is prob the problem"):
the lift still overshot, and a fast reach ended in `power_limit_violation`. The recordings
(`outputs/logs/bus_20261102_170917.lcm`, `bus_20261102_171141.lcm`) show why, and the user's guess
was right:

* The plans' accelerations were bounded per joint (10 rad/s²) but not together. Through `M(q)` the
  feed-forward of a plan change asked for torque steps of 20–49 Nm within 1 ms (libfranka passes
  1 Nm per ms: `kMaxTorqueRate` 1000 Nm/s) and, on the lift far out, 139 Nm of joint 2 (the FR3 has
  87). The torque the robot measured (`tau_J`, which the arm's motion confirms through the URDF
  dynamics) followed the command 20–60 ms late: the lift's braking arrived after the hand had
  passed its goal (2.2 cm overshoot, then down), and on a fast reach the late Kd term pushed
  joints 1 and 3 against each other (their antiphase mode weighs 0.27 kg m² there) at 8 Hz,
  growing to ±2.4 rad/s until the power limit stopped the robot. The simulation had no torque-rate
  limit, so it never showed any of this.
* Fix (one constraint, where the plan is made): the QP bounds the torque each plan asks of the arm,
  `|M â + g + C v| <= 0.8 τ_max` and `|M (â − â_prev)| <= 0.8 τ̇_max Δt` (8 Nm per plan; 0.7 and 7 Nm since round six), as hard
  rows that give way only to the box; the low level ramps the inertial feed-forward at 0.8 τ̇_max
  instead of stepping it, leaving the rest of the robot's torque rate to the Kd term. The
  simulation now has the FR3's torque-rate limit.
* `run_hardware.py` had not noticed this stop either: panda-py's `is_moving()` stays True after a
  reflex; `raise_error()` is the documented check.

## Round five: the lift start (same day)

The user ran it ("Very jerky"). Both lifts shook joints 1 and 3 against each other at 8–9 Hz from
the first 0.1 s of the lift; once it grew into `power_limit_violation`, once it died out. The new
flight recorder had the torque actually sent, and showed:

* The robot's link-side torque follows the command 1–7 ms late and with a 12–38 ms first-order lag
  (joint 1 the slowest). Joints 1 and 3 turning against each other weigh 0.17–0.28 kg m² at the
  lift (they are nearly coaxial), so the per-joint Kd = 50 is a ~300 /s loop there; the lighter
  that motion, the worse the lift went (0.28 settled, 0.20 jerky, 0.17 the reflex).
* With that torque response the simulation reproduces the robot (`configs/sim/env.yaml` now has
  it; without it the simulated arm followed every plan): reach fine, then the shaking at the lift.
* What started it was the plan. At the grasp the bowl rests on the table, so its 2 cm table rows
  were violated; recovering that from rest is infeasible, and the QP's fallback raised the bowl
  the fastest way the box allowed, with the torque rows now on the heavy joints, through the
  light ones at ±10 rad/s², then switched back to the lift field, reversing them. Fix: a held
  point closer than its margin may only not come closer (the lift raises it).
* And plans in general switched joints between ±10 rad/s² from one period to the next (the QP
  riding its box toward a field's demand), which the robot's torque cannot follow. Fix: a jerk
  bound, `|â − â_prev| <= 300 rad/s³ · Δt` (10 rad/s² over the ~30 ms lag), in the same hard
  rows as the torque. In the simulation with the torque response the per-joint Kd was then
  stable from every start pose, where without it it shook from the low ones.
* An inertia-shaped damping (`S M(q) S`, the per-joint Kd on the diagonal) was tried first, with
  the user's go-ahead. It removed the oscillation too, but in light directions it is weak against
  joint friction (a hold stopped 1 cm short) and it tracked worse than the per-joint Kd once the
  plans were realisable. Not kept.
* The lift's top speed is now 0.5 m/s (the reach's): the arm brakes one torque lag after its plan,
  1.6 cm past the goal from 0.8 m/s in the simulation, none from 0.5.

## Round six: the motion's shape (same day)

The user ran it fast and moved the object ("acceleration ... too jerky ... torque limit ... robot goes
red"; "after grasp the arm comes up abruptly and overshoots stiffly and shows shaky motion"). No robot
log in this checkout, so measured in the simulation with `scripts/motion_quality.py` (new: it records
through the settling; `diagnose_tracking.py` stops as `hold` begins). `docs/method.md`, "Smoothness",
has the account. What rounds four and five had not seen:

* The plan itself drove joints 1 and 3 against each other, ±0.5 rad/s on a straight lift: the arm's
  self-motion, priced by nothing but the regulariser. The bounds of rounds four and five limited how
  fast that happened, not that it happened. Priced (`_null_task`), the lift's joint 1 / 3 shake fell
  from 0.068 to 0.002 rad/s, on a change of one QP term.
* The Kd loop was nearly unstable at the fitted torque lag (damping ratio 0.02-0.09). A first pole
  calculation on the bare URDF said 76 % of poses unstable: wrong, it had no rotor inertia.
* The lift was a saturated field, not a trajectory; the reach's posture field stepped to 3 rad/s.
* Wrong turns, kept here so they are not repeated: `task_accel` 5 m/s² doubled the reach from the low
  poses; `null_lambda` 0.15 priced real hand motion and stalled the lift; recomputing the duration from
  the distance left stretched the lift; a metric that smoothed a cut signal reported 45 rad/s²
  of commanded acceleration that never existed.

Not changed: the wrist's Kd, and the sim's torque lags, which remain fitted to one flight recorder.
Validated in simulation only.

## The governing relation

`v <= a H`. A distance row allows approach at `(d − ε)/H`, so braking must deliver `v/H`; with
the hand's acceleration `a` (5–8 m/s² at the FR3 joint limits over the arm's lever) and
`H = 0.10 s`, `v_const = 0.8 m/s` is the bound, and `δ_x >= v²/(2a)` ≈ 6 cm keeps the eq. (5)
decay itself feasible. `H` is now the stopping time, not a latency cover. More speed means more
acceleration or a longer horizon (braking earlier).

## Anti-goals

No thresholds or state rules were added. Everything is a constraint the robot has (acceleration,
torque and torque rate), a
penalty formulation that keeps a constrained problem solvable, the dynamics the paper's PD left out
(inertia), or the gripper's own interface (speed per set-point).

## Measured (real-time sim, `docs/results.md`)

Gold bowl, left-bottom start: grasped 2.7 s after `v`, lifted 0.4 s later (was 4.0 s and 1.3 s).
Lag 2–9 ms per joint. Gripper stream 2 % stopped (was 33 %). The evaluation batches of the Table I
set, Fig. 7, Fig. 8 and carried objects are in `docs/results.md`.

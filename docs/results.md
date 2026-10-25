# Simulation results (FR3 + 2F-85, MuJoCo)

The paper's simulation studies (Sec. III-A: Table I, Figs. 6-8) repeated with
this implementation on the Franka FR3 + Robotiq 2F-85. The paper's robot is a
15-DoF arm-hand; the numbers below are ours on the FR3, not the paper's, and
the paper's are quoted for comparison only.

Everything here comes from `scripts/eval_sim.py` on a MacBook (Apple
Silicon), the real-time simulation running alone on the machine (one trial at
a time: `run_env.py` + `run_controller.py` over LCM, the planner at 100 Hz,
the control loop at 1 kHz, 20 s per trial as in the paper, 30 s for the
disturbance and obstacle studies). The commands:

```bash
python scripts/eval_sim.py --objects box blue_bowl blue_mug wine_glass --orientations all \
    --seeds 5 --fields path linear --out outputs/eval/table1                         # Table I, 360 trials
python scripts/eval_sim.py --objects blue_bowl blue_mug --poses center_up --seeds 3 \
    --push_at 1.5 4.0 --jump_at 3.0 --timeout 30 --out outputs/eval/fig7              # Fig. 7
python scripts/eval_sim.py --env configs/sim/env_obstacles.yaml --objects wine_glass blue_bowl \
    --seeds 3 --timeout 30 --out outputs/eval/fig8                                     # Fig. 8
```

Each trial directory holds the env / controller logs and `trace.npz`
(fingertip and target centres over time for `scripts/plot_trace.py`; the
measured state for `scripts/dissect.py`). A trial counts as a success when
the object is held (`is_grasped`) 5 cm above its rest height within the time
limit; trials where the bus or a process died are marked `infra` and
repeated, never counted.

## Table I: success rates

Paper's protocol: three start poses (left-bottom, center-up, right-bottom),
five placements each; the mug and the wine glass in five orientations
(upright, ±90° about x, ±90° about y), the others upright. `path` is the
method (eqs. 3-5 fingertip fields), `linear` the eq. (2) baseline, both
through the same eq. (11) QP.

| Object | Trials | **path** (this work, FR3) | **linear** (this work, FR3) | paper "Ours" | paper "Linear" |
|---|---|---|---|---|---|
| Box | 15 | **100.0 %** (15/15) | 100.0 % (15/15) | 100.0 % | 100.0 % |
| Bowl | 15 | **100.0 %** (15/15) | 100.0 % (15/15) | 100.0 % | 33.3 % |
| Mug | 75 | **97.3 %** (73/75) | 73.3 % (55/75) | 100.0 % | 30.7 % |
| Wine glass | 75 | **98.7 %** (74/75) | 85.3 % (64/75) | 100.0 % | 32.0 % |
| Dish | - | not run (no 2F-85 grasp satisfies the paper's margins, docs/method.md) | | 100.0 % | 93.3 % |
| **All** | 180 | **98.3 %** (177/180) | 82.8 % (149/180) | | |

By orientation (successes / 15, the three start poses × five placements):

| | upright | rx+90 | rx−90 | ry+90 | ry−90 |
|---|---|---|---|---|---|
| Mug, path | 15 | 15 | **13** | 15 | 15 |
| Mug, linear | 9 | 12 | 8 | 13 | 13 |
| Wine glass, path | 15 | 15 | 15 | 15 | **14** |
| Wine glass, linear | 15 | 13 | 12 | 13 | 11 |

Time from `v` to the object lifted: median 4.6 s (path), 7.5 s (linear).
Planning time per cycle: median 7.9 ms (path: eq. 3 via search + 3 SQP
iterations + eq. 11), 2.2 ms (linear), 12 ms with the four moving obstacles,
all well inside the 10 ms planning period. Control-loop rates logged per
trial: joint measurements 972-1000 Hz, commands 696-800 Hz (the Python
control thread on the laptop; on the robot libfranka's 1 kHz applies).

**Why our linear baseline is stronger than the paper's.** The 2F-85's two
pads are a rigid pair 85 mm apart at most, and its rim and handle targets
are 1.5 cm off the surface; sliding a parallel jaw down over a rim reaches
most of them without any detour, so the eq. (2) attractor succeeds on the
bowl always and on the upright wine glass always. Where it fails it fails
the way the paper's Fig. 6 shows: trapped against the object. Every one of
its 31 failures ends with hand spheres at their 1 cm object margin
(`scripts/rollout.py --fields linear` prints what binds): the finger pushed
into the mug's body or handle, or the glass's bowl, with the fingertips
0.3-6 cm from their targets. Since the paper's own fingertips are
independent, its linear baseline has more ways to get stuck.

## Fig. 6: fingertip error at convergence

Centre error of the two fingertips to the selected pair when the gripper
closed (path field, 177 successes): median 0.35 cm, 90th percentile 0.55 cm,
maximum 0.88 cm; the closing trigger is 1 cm per fingertip (paper: 1 cm).
Linear, 149 successes: median 0.37 cm, maximum 0.95 cm. Linear failures:
median 1.1 cm, up to 6.1 cm (above). The paper's "mostly below 1 cm" for its
method, and its linear failures spread above the threshold, are what we see.

## Fig. 7: disturbances

Six trials (bowl and mug, center-up start, three placements): a push on the
hand at 1.5 s (a random horizontal 80 N force for 0.3 s), the object moved
to a new random pose at 3 s, a second push at 4 s. **5 of 6 lifted** the
object (bowl 3/3 in 6.1-7.1 s, mug 2/3). The traces
(`outputs/eval/fig7/*/trace.png`, like the paper's Fig. 7) show the
fingertip centre pushed away and returning, the target curves jumping when
the object moves, and the re-convergence; the target also changes slightly
during the pushes (eq. 12 re-selects as the hand moves), as the paper notes.

The failed mug trial (seed 2) is not the reach failing to recover: the
second push drove the hand into the mug and knocked it over (the recorded
object height goes from 0 to 3.4 cm, its axis to 45°). The controller
re-targeted the lying mug at once (`trace.png`: the target z drops at 4.3 s)
and brought the fingertips to 1.1 cm and 0.3 cm of the new targets, then
stalled: the fingertip sphere and a linkage sphere at their object margin and
joint 5 at its limit. Replaying the recorded state kinematically
(`scripts/dissect.py <trial>/trace.npz --t_start 4.5`) stalls at the same
place, so it is the method's local minimum on that pose, not the dynamics.
Deterministic, documented below with the Table I stalls.

## Fig. 8: moving obstacles

Four boxes oscillating around the object (`configs/sim/env_obstacles.yaml`,
exact box distance functions, 5 cm margin in the paths, 2 cm in the QP),
wine glass and bowl, three start poses × three placements: **18 of 18
lifted, no robot-obstacle contact in any trial** (the env counts contacts
between robot and obstacle geometry every step). One trial
(bowl, center-up, seed 3) was lost to a network outage that took the LCM
multicast route down mid-trial; it was repeated alone afterwards
(`outputs/eval/rerun_fig8`, 3/3) and `eval_sim.py` now marks a trial whose
streams stop as `infra` and retries it.

## The failures, dissected

All four non-infra failures of the method (three in Table I, one in Fig. 7)
reproduce exactly in a kinematic replay with perfect tracking
(`scripts/rollout.py` / `scripts/dissect.py`), so they are properties of the
method on this robot, not of the simulation's dynamics or timing:

| Trial | Object pose | Short by | What binds at the end |
|---|---|---|---|
| mug rx−90, left-bottom, seed 2 | lying, 48 cm ahead, 4 cm right of the base | 0.5 / 1.1 cm | joint 4 at −3.04 rad (its limit); link6-link1 at the 2 cm self-collision margin; finger, linkage and palm spheres at the 1 cm object margin |
| mug rx−90, left-bottom, seed 3 | lying, 46 cm / 5 cm right | 0.8 / 1.4 cm | the same set |
| wine glass ry−90, right-bottom, seed 3 | lying, 46 cm / 2 cm right | 2.1 / 0.7 cm | joint 4 at its limit; link6-link1 and link5-link2 at their margins; finger spheres at the object margin |
| mug, Fig. 7 seed 2 (after the knock-over) | lying tilted, 39 cm / 13 cm right | 1.1 / 0.3 cm | joint 5 at its limit; fingertip and linkage spheres at the object margin |

The pattern is the one the paper describes in Fig. 12 and Sec. IV: the
fingertip targets are planned without the joint configuration, and when the
object lies on its side close to the base the FR3 has to fold its elbow
fully to bring the pads in horizontally; the joint limit, the self-collision
margin and the hand's object margin become active together and the QP's
best feasible motion is zero, 1-2 cm from the trigger. The paper's remedy
(velocity targets for the elbow, Sec. IV) is future work there as here; no
such term has been added. Every plan reports this set in
`ReachStep.info['binding']` and the controller prints it with its periodic
plan-time line, so the same question can be answered on the robot.

## Videos

`outputs/videos/` (`scripts/record_video.py`, offscreen MuJoCo with the
fingertips, their targets and the planned paths drawn in, close-up inset):

1. `1_mug_path_vs_linear.mp4`: the lying mug from the left-bottom start, the path field (lifts) beside the linear field (stuck against the mug).
2. `2_bowl_from_above.mp4`: the bowl, rim pinch from the center-up start.
3. `3_wine_glass_on_its_side.mp4`: the wine glass lying on its side.
4. `4_disturbances_pushes_and_moved_object.mp4`: pushes on the hand and the object moved mid-reach (Fig. 7).
5. `5_moving_obstacles.mp4`: the wine glass among the four moving boxes (Fig. 8).

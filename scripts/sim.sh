#!/usr/bin/env bash
# The simulation in three tmux panes: env (MuJoCo viewer), controller, browser view.
#   scripts/sim.sh [object] [field] [extra env config]
#   scripts/sim.sh blue_mug linear
#   scripts/sim.sh wine_glass path configs/sim/env_obstacles.yaml
# In the controller pane: v = reach + grasp, x = open, h = home, q = quit.
# In the env pane / viewer: k = push the hand, o = move the object.
set -euo pipefail
cd "$(dirname "$0")/.."
OBJ=${1:-blue_bowl}
FIELD=${2:-path}
EXTRA=${3:-}
PY=${PYTHON:-python}
command -v tmux >/dev/null || { echo "needs tmux (or run the three commands below in three terminals)"; exit 1; }
S=hrg
tmux kill-session -t $S 2>/dev/null || true
tmux new-session -d -s $S "$PY run_env.py --config configs/sim/env.yaml $EXTRA --object_config configs/objects/$OBJ.yaml; read"
tmux split-window -h -t $S "sleep 4; $PY run_controller.py --config configs/controller.yaml --object_config configs/objects/$OBJ.yaml --hrg.field $FIELD; read"
tmux split-window -v -t $S "$PY run_visualizer.py --config configs/visualizer.yaml --object_mesh assets/objects/$OBJ/visual.obj; read"
tmux select-pane -t $S:0.1
tmux attach -t $S

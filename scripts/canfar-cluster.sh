#!/bin/sh
# scripts/canfar-cluster.sh
#
# Start a Ray cluster sized to a Slurm allocation on CANFAR and run a
# command inside it.  The cluster scales to the allocation: `--nodes N`
# gives you exactly N compute nodes (head + N-1 workers); CANFAR is Slurm,
# not Kubernetes, so there is no KubeRay/elastic autoscaler — every task
# of a ray-union run is submitted up front, and Ray spreads them across
# the nodes automatically.
#
# Usage (from a login node — this submits itself):
#   scripts/canfar-cluster.sh --nodes 4 \
#       --command "pixi run xmatch match gaia desils --union -o full.hats --retries 2"
#
# or submit the script directly:
#   sbatch scripts/canfar-cluster.sh --nodes 4 --time 12:00:00 --account def-x \
#       --command "pixi run xmatch match gaia desils --union -o full.hats --retries 2"
#
# Inside the allocation: the batch script runs exactly once (on the first
# node), so it fans itself out with `srun` — one instance per compute
# node.  Rank 0 (SLURM_PROCID=0) runs `ray start --head`, the other ranks
# join with `ray start --address=$HEAD:6379`, rank 0 waits until every
# node reports in (120 s cap), exports RAY_ADDRESS=$HEAD:6379 and runs
# CMD with that environment, and every rank stops its Ray runtime on exit
# (trap), so the job never leaves stray raylets on the nodes.
#
# IMPORTANT — the union cache must live on storage shared by all compute
# nodes: point XMATCH_CACHE_ROOT / --cache-root at a shared path (e.g.
# /scratch/... or a project dir), or use a `vos:` root with `cache.roots`
# replicas.  Node-local /scratch breaks remote workers' reads.
#
# Options:
#   --nodes N       compute nodes for the allocation (default 4)
#   --account ACCT  Slurm account (optional)
#   --time HH:MM:SS wall time (default 06:00:00)
#   --command CMD   command to run inside the cluster (shell string)
#   --dry-run       print exactly what would run; never submits or starts Ray

set -eu

NODES=4
ACCOUNT=""
TIME="06:00:00"
CMD='pixi run xmatch match gaia desils --union -o full.hats --retries 2'
DRY_RUN=0
SRUN_CHILD=0

usage() {
    sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --nodes)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            NODES="$2"; shift 2 ;;
        --account)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            ACCOUNT="$2"; shift 2 ;;
        --time)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            TIME="$2"; shift 2 ;;
        --command)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            CMD="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --srun-child)
            SRUN_CHILD=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# shell-quote a string for display (single quotes, POSIX-safe)
shq() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

# ---------------------------------------------------------------- submit
if [ -z "${SLURM_JOB_ID:-}" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        acct_disp=""
        [ -n "$ACCOUNT" ] && acct_disp=" --account=$ACCOUNT"
        echo "== CANFAR cluster dry run =="
        echo "From a login node this would submit:"
        echo "  sbatch --nodes=$NODES --ntasks=$NODES --time=$TIME$acct_disp"
        echo "         --parsable $0 --command $(shq "$CMD")"
        echo
        echo "Inside the allocation the batch script (one run, first node) fans out:"
        echo "  srun --nodes=$NODES --ntasks=$NODES --ntasks-per-node=1"
        echo "       $0 --command $(shq "$CMD") --srun-child"
        echo
        echo "Rank 0 would run:"
        echo "  pixi run ray start --head --port=6379 --dashboard-host=127.0.0.1 --num-cpus=\$(nproc)"
        echo "  (head hostname scraped from 'ray start' output -> \$HEAD)"
        echo "  export RAY_ADDRESS=\$HEAD:6379"
        echo "  bash -lc $(shq "$CMD")"
        echo "  pixi run ray stop --force   (every rank, via trap EXIT)"
        echo
        echo "Every other rank would run:"
        echo "  pixi run ray start --address=\$HEAD:6379 --num-cpus=\$(nproc)"
        echo "  (then wait for the head to finish and stop its ray)"
        exit 0
    fi
    set -- sbatch --nodes="$NODES" --ntasks="$NODES" --time="$TIME"
    [ -n "$ACCOUNT" ] && set -- "$@" --account="$ACCOUNT"
    set -- "$@" --parsable "$0" --command "$CMD"
    out=$("$@")
    echo "Submitted batch job $out"
    squeue -u "${USER:-}" >/dev/null 2>&1 || true
    exit 0
fi

# ---------------------------------------------------------------- allocation
# sbatch runs the batch script ONCE (on the first node; --ntasks only
# reserves tasks).  To get one instance per compute node we re-launch
# ourselves through srun: each srun task gets a distinct SLURM_PROCID and
# lands on its own node.  `--srun-child` marks the re-launched instances
# so they don't fan out again.
HEAD_FILE="${SLURM_JOB_ID}.ray-head"
EXPECTED_NODES="${SLURM_NNODES:-$NODES}"

if [ "$SRUN_CHILD" -eq 0 ]; then
    srun --nodes="$EXPECTED_NODES" --ntasks="$EXPECTED_NODES" \
        --ntasks-per-node=1 "$0" --command "$CMD" --srun-child
    exit $?
fi

cleanup() {
    pixi run ray stop --force >/dev/null 2>&1 || true
    rm -f "$HEAD_FILE" 2>/dev/null || true
}
trap cleanup EXIT

# Wall-time budget for the worker-side wait: allocation time plus margin.
_h=${TIME%%:*}; _rest=${TIME#*:}; _m=${_rest%%:*}; _s=${_rest##*:}
WALL_SEC=$((_h * 3600 + _m * 60 + _s + 600))

if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
    # ---- head: start the cluster, wait for every node, run CMD
    start_out=$(pixi run ray start --head --port=6379 --dashboard-host=127.0.0.1 \
        --num-cpus="$(nproc)" 2>&1)
    # Ray 2.x head prints the join line `ray start --address='<ip>:6379'`
    HEAD=$(printf '%s\n' "$start_out" | sed -n "s/.*--address='\([^:']*\):.*/\1/p" | head -n 1)
    if [ -z "$HEAD" ]; then
        HEAD=$(hostname)
    fi
    printf '%s\n' "$HEAD" > "$HEAD_FILE"

    # ray status lists healthy nodes as `N node_<hex>` lines (no "raylet"
    # text in Ray 2.x); count those until every node is in.
    i=0
    n=0
    while [ "$i" -lt 120 ]; do
        n=$(pixi run ray status 2>/dev/null | grep -c 'node_' || true)
        [ "$n" -ge "$EXPECTED_NODES" ] && break
        i=$((i + 1))
        sleep 1
    done
    if [ "$n" -lt "$EXPECTED_NODES" ]; then
        echo "ray cluster reached $n of $EXPECTED_NODES node(s) after 120 s; aborting" >&2
        exit 1
    fi
    echo "ray cluster up: $n node(s)"
    echo "export RAY_ADDRESS=$HEAD:6379"
    RAY_ADDRESS="$HEAD:6379"
    export RAY_ADDRESS
    bash -lc "$CMD"
    exit 0
fi

# ---- worker: join the head, then wait for the job to finish
i=0
while [ ! -s "$HEAD_FILE" ]; do
    if [ "$i" -ge 120 ]; then
        echo "timed out waiting for $HEAD_FILE on $(hostname)" >&2
        exit 1
    fi
    i=$((i + 1))
    sleep 1
done
HEAD=$(cat "$HEAD_FILE")
pixi run ray start --address="$HEAD:6379" --num-cpus="$(nproc)"
i=0
while [ -f "$HEAD_FILE" ]; do
    i=$((i + 5))
    if [ "$i" -ge "$WALL_SEC" ]; then
        echo "worker $(hostname): head did not finish within the wall budget; exiting" >&2
        exit 1
    fi
    sleep 5
done
exit 0

#!/bin/sh
# scripts/canfar-ray-job.sh
#
# Run an xmatch command as a Ray Job on the CANFAR science platform
# (https://www.opencadc.org/canfar/ — VOSpace + Skaha container sessions;
# there is no Slurm).  Ray runs as a contributed `ray-manager` Skaha
# session (the head) plus headless `ray-worker` sessions (the workers);
# this script is a thin driver over the platform CLIs:
#
#   1. preflight — `canfar` client present and authenticated (`canfar ps`)
#   2. cluster   — `astroai cluster start` configures the ray-manager and
#                  launches N worker sessions (skipped when
#                  CANFAR_RAY_JOBS_ADDRESS / --address is already set)
#   3. submit    — `astroai jobs submit --cmd "$CMD" --wait` runs the
#                  command on the Ray Jobs API; the script's exit code is
#                  the job's
#
# The ray-manager session itself is started from the AstroAI hub ("Start
# batch compute") or with:
#   canfar create --cpu 2 --memory 8 --name xmatch-ray contributed images.canfar.net/astroai/ray-manager:<tag>
# (or pass --create-manager IMAGE to have this script create it first).
#
# Usage (from any AstroAI/CANFAR session with the platform CLIs on PATH):
#   scripts/canfar-ray-job.sh \
#       --command "pixi run python scripts/bench_ray_union.py --rows 100000 --repeat 1 --warmup 0 --work-dir /arc/projects/hats/xmatch-bench-unique"
#
# IMPORTANT — the union cache must be visible to every worker pod.  On the
# platform the defaults are already shared: the mirrored-HATS cache root is
# /arc/projects/hats (inputs) and bare relative outputs land in
# /arc/projects/hats/xmatch — never a pod-local home.  Override with
# `--env XMATCH_CACHE_ROOT=vos:hats` (VOSpace root) or XMATCH_OUTPUT_ROOT=...
# and `-o vos:hats/xmatch/name.hats` works too (staged locally, uploaded;
# resume pulls the remote tree back first).  Bare output names share the
# root: use a distinct name per run — a rerun with different parameters
# wipes the previous run's stale chunks.  CANFAR `/scratch` is per-pod and
# breaks remote workers' reads.
#
# Options:
#   --command CMD        required command to run as the Ray Job (shell string)
#   --workers N          min/max ray-worker sessions to keep (default 4)
#   --cores N            CPUs per worker (default 1)
#   --ram GiB            RAM per worker (default 4)
#   --cpus N             entrypoint CPUs for the job (default 2)
#   --memory GiB         entrypoint memory reservation (optional)
#   --env KEY=VALUE      environment for the job (repeatable)
#   --cwd DIR            job working directory, uploaded to the head
#                        (non-ignored files, including untracked files —
#                        inspect it for secrets; .gitignore applies).
#                        Default: none — on the platform (webterm, /arc
#                        mounted) the command is self-located into the
#                        current directory instead, so nothing uploads,
#                        the installed pixi env is reused and outputs
#                        land in the repo.  From a laptop, self-locate:
#                        --command "bash -lc 'cd /arc/... && pixi run ...'"
#   --create-manager IMG create a 2-CPU/8-GiB ray-manager session first
#   --manager-name NAME  manager session name (default xmatch-ray)
#   --manager URL        manager connect URL (cluster start --address)
#   --address URL        Jobs API URL (skip cluster start)
#   --dry-run            print exactly what would run; runs nothing

set -eu

WORKERS=4
CORES=1
RAM=4
CPUS=2
MEMORY=""
CMD=""
ENVS=""
CWD=""
CREATE_MANAGER=""
MGR_NAME="xmatch-ray"
MANAGER=""
ADDRESS=""
DRY_RUN=0

usage() {
    sed -n '2,/^set -eu$/p' "$0" | sed -e '/^set -eu$/d' -e 's/^# \{0,1\}//'
}

# shell-quote a string for display / safe eval (single quotes, POSIX-safe)
shq() {
    printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --command)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            CMD="$2"; shift 2 ;;
        --workers)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            WORKERS="$2"; shift 2 ;;
        --cores)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            CORES="$2"; shift 2 ;;
        --ram)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            RAM="$2"; shift 2 ;;
        --cpus)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            CPUS="$2"; shift 2 ;;
        --memory)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            MEMORY="$2"; shift 2 ;;
        --env)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            ENVS="$ENVS --env $(shq "$2")"; shift 2 ;;
        --cwd)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            CWD="$2"; shift 2 ;;
        --create-manager)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            CREATE_MANAGER="$2"; shift 2 ;;
        --manager-name)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            MGR_NAME="$2"; shift 2 ;;
        --manager)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            MANAGER="$2"; shift 2 ;;
        --address)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            ADDRESS="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [ -z "$CMD" ]; then
    echo "error: --command is required; pass a bounded workload explicitly" >&2
    usage
    exit 2
fi

# Job working directory.  On the platform (webterm; /arc mounted) the repo
# at $(pwd) already lives on the head pod via the shared /arc volume — so
# default to SELF-LOCATING the command (`bash -c 'cd <pwd> && CMD'`): zero
# upload, the installed pixi env is reused, and outputs land in the repo.
# Off-platform (laptop) there is no safe default: the user passes --cwd
# (explicit upload) or a self-locating command.
SELF_LOCATE=""
if [ -z "$CWD" ]; then
    if [ -f pixi.toml ] && [ -d /arc ]; then
        SELF_LOCATE="bash -c $(shq "cd $(pwd) && $CMD")"
    else
        echo "note: no self-locating default (needs pixi.toml + the /arc mount);" >&2
        echo "  the job would run in the head's default working directory." >&2
        echo "  Run this from the repo on the manager, pass --cwd (uploads this" >&2
        echo "  copy's non-ignored files, including untracked files; inspect it" >&2
        echo "  for secrets), or self-locate:" >&2
        echo "  --command \"bash -lc 'cd /abs/manager/path && pixi run ...'\"" >&2
    fi
elif [ ! -f "$CWD/.gitignore" ]; then
    echo "note: --cwd $CWD has no .gitignore — Ray uploads everything in it" >&2
    echo "  (including secrets). Add a .gitignore or use a self-locating command." >&2
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "== CANFAR ray-job dry run =="
    echo "preflight:"
    echo "  canfar ps >/dev/null"
    if [ -n "$CREATE_MANAGER" ]; then
        echo "  canfar create --cpu 2 --memory 8 --name $MGR_NAME contributed $(shq "$CREATE_MANAGER")"
    fi
    echo "cluster:"
    if [ -n "$ADDRESS" ]; then
        echo "  (skipped — using --address $(shq "$ADDRESS"))"
    elif [ -n "${CANFAR_RAY_JOBS_ADDRESS:-}" ]; then
        echo "  (skipped — CANFAR_RAY_JOBS_ADDRESS already set)"
    else
        echo "  astroai cluster start --min-workers $WORKERS --max-workers $WORKERS --cores $CORES --ram $RAM${MANAGER:+ --address $(shq "$MANAGER")}"
    fi
    RUN_CMD="$CMD"
    [ -n "$SELF_LOCATE" ] && RUN_CMD="$SELF_LOCATE"
    ADDR_DISP="${ADDRESS:-${CANFAR_RAY_JOBS_ADDRESS:-<resolved by cluster start>}}"
    echo "submit:"
    echo "  astroai jobs submit --cmd $(shq "$RUN_CMD") --cpus $CPUS --address $(shq "$ADDR_DISP")${MEMORY:+ --memory $(shq "$MEMORY")}${CWD:+ --cwd $(shq "$CWD")} --wait$ENVS"
    exit 0
fi

# ---------------------------------------------------------------- preflight
command -v canfar >/dev/null 2>&1 || {
    echo "error: 'canfar' client not found on PATH" >&2
    echo "  run this from a CANFAR/AstroAI session (webterm), or install the" >&2
    echo "  canfar client (PyPI 'canfar') and log in with 'canfar login'" >&2
    exit 1
}
command -v astroai >/dev/null 2>&1 || {
    echo "error: 'astroai' not found on PATH" >&2
    echo "  run this from an AstroAI session with canfar-lab installed" >&2
    exit 1
}
canfar ps >/dev/null 2>&1 || {
    echo "error: cannot list sessions — are you logged in? (canfar login)" >&2
    exit 1
}

# ---------------------------------------------------------------- cluster
if [ -z "$ADDRESS" ] && [ -z "${CANFAR_RAY_JOBS_ADDRESS:-}" ]; then
    if [ -n "$CREATE_MANAGER" ]; then
        echo "creating ray-manager session '$MGR_NAME' from $CREATE_MANAGER"
        canfar create --cpu 2 --memory 8 --name "$MGR_NAME" contributed "$CREATE_MANAGER"
    fi
    echo "starting ray cluster ($WORKERS worker(s), ${CORES}c/${RAM}GiB each)"
    set -- astroai cluster start --min-workers "$WORKERS" --max-workers "$WORKERS" \
        --cores "$CORES" --ram "$RAM"
    [ -n "$MANAGER" ] && set -- "$@" --address "$MANAGER"
    start_out=$("$@" 2>&1) || {
        echo "$start_out" >&2
        echo "error: cluster start failed — is a ray-manager session running?" >&2
        echo "  start one from the AstroAI hub, or: canfar create --cpu 2 --memory 8 --name $MGR_NAME" >&2
        echo "  contributed images.canfar.net/astroai/ray-manager:<tag>" >&2
        exit 1
    }
    ADDRESS=$(printf '%s\n' "$start_out" | sed -n 's/^export CANFAR_RAY_JOBS_ADDRESS=//p' | tail -n 1)
    if [ -z "$ADDRESS" ]; then
        echo "$start_out" >&2
        echo "error: could not read the Jobs address from 'cluster start' output" >&2
        exit 1
    fi
    echo "$start_out"
else
    ADDRESS="${ADDRESS:-${CANFAR_RAY_JOBS_ADDRESS:-}}"
    echo "using jobs address: $ADDRESS"
fi

# ---------------------------------------------------------------- submit
RUN_CMD="$CMD"
[ -n "$SELF_LOCATE" ] && RUN_CMD="$SELF_LOCATE"
set -- astroai jobs submit --cmd "$RUN_CMD" --cpus "$CPUS" \
    --address "$ADDRESS" --wait
[ -n "$MEMORY" ] && set -- "$@" --memory "$MEMORY"
[ -n "$CWD" ] && set -- "$@" --cwd "$CWD"
eval "set -- \"\$@\"$ENVS"
echo "submitting as a Ray job: $RUN_CMD"
"$@"
exit $?

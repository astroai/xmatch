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
#   2. cluster   — `astroai-workload cluster ensure` launches N worker
#                  sessions through the ray-manager (idempotent; skipped
#                  when ASTROAI_RAY_JOBS_ADDRESS / --address is already set)
#   3. submit    — `astroai-workload submit --cmd "$CMD" --wait` runs the
#                  command on the Ray Jobs API; the script's exit code is
#                  the job's
#
# The ray-manager session itself is started from the AstroAI hub ("Start
# batch compute") or with:
#   canfar create --name xmatch-ray contributed images.canfar.net/astroai/ray-manager:<tag>
# (or pass --create-manager IMAGE to have this script create it first).
#
# Usage (from any AstroAI/CANFAR session with the platform CLIs on PATH):
#   scripts/canfar-ray-job.sh \
#       --command "pixi run xmatch match gaia desils --union -o full.hats --retries 2"
#
# IMPORTANT — the union cache must be visible to every worker pod: point
# XMATCH_CACHE_ROOT / --cache-root at VOSpace (`vos:...`) or a shared
# /arc path (pass `--env XMATCH_CACHE_ROOT=vos:xmatch-cache`); CANFAR
# `/scratch` is per-pod and breaks remote workers' reads.
#
# Options:
#   --command CMD        command to run as the Ray Job (shell string)
#   --workers N          ray-worker sessions to launch (default 4)
#   --cores N            CPUs per worker (default 1)
#   --ram GiB            RAM per worker (default 4)
#   --cpus N             entrypoint CPUs for the job (default 4)
#   --memory GiB         entrypoint memory reservation (optional)
#   --env KEY=VALUE      environment for the job (repeatable)
#   --cwd DIR            job working directory; default: the current
#                        directory when it holds a pixi.toml (the repo on
#                        the manager).  Ray uploads --cwd to the head
#                        (tracked files only — .gitignore is respected,
#                        including nested files, so .pixi/ is skipped);
#                        from a laptop, self-locate instead:
#                        --command "bash -lc 'cd /arc/... && pixi run ...'"
#   --create-manager IMG create the ray-manager session first
#   --manager-name NAME  manager session name (default xmatch-ray)
#   --manager URL        manager connect URL (cluster ensure --address)
#   --address URL        Jobs API URL (skip cluster ensure)
#   --dry-run            print exactly what would run; runs nothing

set -eu

WORKERS=4
CORES=1
RAM=4
CPUS=2
MEMORY=""
CMD='pixi run xmatch match gaia desils --union -o full.hats --retries 2'
ENVS=""
CWD=""
CREATE_MANAGER=""
MGR_NAME="xmatch-ray"
MANAGER=""
ADDRESS=""
DRY_RUN=0

usage() {
    sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'
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

# Job working directory: default to the current directory when it looks
# like the project (pixi.toml present) — Ray uploads it to the head,
# respecting .gitignore (nested too, so .pixi/ is never packaged).
if [ -z "$CWD" ] && [ -f pixi.toml ]; then
    CWD="$(pwd)"
elif [ -z "$CWD" ]; then
    echo "note: no pixi.toml in $(pwd); the job runs in the head's default" >&2
    echo "  working directory.  Run this from the repo on the manager, pass" >&2
    echo "  --cwd, or self-locate with: --command \"bash -lc 'cd /abs && pixi run ...'\"" >&2
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "== CANFAR ray-job dry run =="
    echo "preflight:"
    echo "  canfar ps >/dev/null"
    if [ -n "$CREATE_MANAGER" ]; then
        echo "  canfar create --name $MGR_NAME contributed $(shq "$CREATE_MANAGER")"
    fi
    echo "cluster:"
    if [ -n "$ADDRESS" ]; then
        echo "  (skipped — using --address $(shq "$ADDRESS"))"
    elif [ -n "${ASTROAI_RAY_JOBS_ADDRESS:-}" ]; then
        echo "  (skipped — ASTROAI_RAY_JOBS_ADDRESS already set)"
    else
        echo "  astroai-workload cluster ensure --workers $WORKERS --cores $CORES --ram $RAM${MANAGER:+ --address $(shq "$MANAGER")}"
    fi
    echo "submit:"
    echo "  astroai-workload submit --cmd $(shq "$CMD") --cpus $CPUS${MEMORY:+ --memory $(shq "$MEMORY")}${CWD:+ --cwd $(shq "$CWD")} --wait$ENVS"
    exit 0
fi

# ---------------------------------------------------------------- preflight
command -v canfar >/dev/null 2>&1 || {
    echo "error: 'canfar' client not found on PATH" >&2
    echo "  run this from a CANFAR/AstroAI session (webterm), or install the" >&2
    echo "  canfar client (PyPI 'canfar') and log in with 'canfar login'" >&2
    exit 1
}
command -v astroai-workload >/dev/null 2>&1 || {
    echo "error: 'astroai-workload' not found on PATH" >&2
    echo "  it ships in the ray-manager/ray-worker images and AstroAI" >&2
    echo "  sessions; standalone: pip install astroai-workload" >&2
    exit 1
}
canfar ps >/dev/null 2>&1 || {
    echo "error: cannot list sessions — are you logged in? (canfar login)" >&2
    exit 1
}

# ---------------------------------------------------------------- cluster
if [ -z "$ADDRESS" ] && [ -z "${ASTROAI_RAY_JOBS_ADDRESS:-}" ]; then
    if [ -n "$CREATE_MANAGER" ]; then
        echo "creating ray-manager session '$MGR_NAME' from $CREATE_MANAGER"
        canfar create --name "$MGR_NAME" contributed "$CREATE_MANAGER"
    fi
    echo "ensuring ray cluster ($WORKERS worker(s), ${CORES}c/${RAM}GiB each)"
    set -- astroai-workload cluster ensure --workers "$WORKERS" \
        --cores "$CORES" --ram "$RAM"
    [ -n "$MANAGER" ] && set -- "$@" --address "$MANAGER"
    ensure_out=$("$@" 2>&1) || {
        echo "$ensure_out" >&2
        echo "error: cluster ensure failed — is a ray-manager session running?" >&2
        echo "  start one from the AstroAI hub, or: canfar create --name $MGR_NAME" >&2
        echo "  contributed images.canfar.net/astroai/ray-manager:<tag>" >&2
        exit 1
    }
    ADDRESS=$(printf '%s\n' "$ensure_out" | sed -n 's/^export ASTROAI_RAY_JOBS_ADDRESS=//p' | tail -n 1)
    if [ -z "$ADDRESS" ]; then
        echo "$ensure_out" >&2
        echo "error: could not read the Jobs address from 'cluster ensure' output" >&2
        exit 1
    fi
    echo "$ensure_out"
else
    ADDRESS="${ADDRESS:-${ASTROAI_RAY_JOBS_ADDRESS}}"
    echo "using jobs address: $ADDRESS"
fi

# ---------------------------------------------------------------- submit
set -- astroai-workload submit --cmd "$CMD" --cpus "$CPUS" \
    --address "$ADDRESS" --wait
[ -n "$MEMORY" ] && set -- "$@" --memory "$MEMORY"
[ -n "$CWD" ] && set -- "$@" --cwd "$CWD"
eval "set -- \"\$@\"$ENVS"
echo "submitting as a Ray job: $CMD"
"$@"
exit $?

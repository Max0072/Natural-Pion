#!/bin/bash
# Copy the C4 corpus from $DATA_p330 (NFS) to /tmp (local disk) once per node,
# and print the local paths to read the data from instead.
#
# WHY THIS EXISTS (2026-09-22). $DATA_p330 is NFS4 over RDMA from one server,
# `hard`-mounted -- a client whose server is slow does not error, it waits
# forever, which is exactly the "0% GPU, process alive, nothing happening"
# signature this project hit twice in one evening (docs/CLUSTER.md, "rtx6002
# wedges too"). The mount also carries `fsc` (FS-Cache, a local-disk cache
# layer for NFS reads), and on this node's `systemctl status cachefilesd` the
# daemon that backs it has been dead since 2026-09-08 (`SIGSEGV`, core-dumped,
# 14 days inactive at the time this was found) -- a root-owned system service
# this project cannot restart. Report it to cluster admins; it is likely a
# one-command fix for them and was invisible until someone checked.
#
# Until that is fixed -- and to be safe even after it is, since a shared NFS
# server can degrade for other reasons -- this script moves the *hot, heavily
# re-read, unchanging* corpus off NFS entirely. `/tmp` (`zlocal/tmp`, ZFS) is
# genuinely local: not NFS, not autofs, unaffected by the server at
# 172.31.0.14/.12 that backs $DATA_p330 and $DATA_p330/../scratch alike (both
# are `hard`-mounted NFS with the same dead-cache dependency; scratch is not a
# safe alternative for this).
#
# WHAT IT DOES NOT FIX. Run directories, checkpoints and logs still live on
# $DATA_p330/runs -- that traffic is much smaller (13 MB of manifest+log per
# run, checkpoints only at save time) and less latency-sensitive than the
# scattered per-step reads a training loop makes against the corpus, so it is
# not staged here. If a wedge recurs with the corpus already local, that is
# new information: it would point at the run-directory writes or something
# else entirely, not at this.
#
# USAGE
#   source scripts/stage_corpus.sh              # uses $DATA_p330/c4, /tmp/c4
#   TRAIN_PATH=... VAL_PATH=...                  # exported for the caller
#
# Safe under concurrent invocation: an flock serialises stagers on one node,
# so an 8-task array does not copy the corpus eight times at once, and a task
# that finds it already staged (exact byte size match) returns in well under a
# second.
set -uo pipefail

_stage_src="${1:-$DATA_p330/c4}"
_stage_dst="${2:-/tmp/c4}"

mkdir -p "$_stage_dst"
exec 9>"$_stage_dst/.stage.lock"
flock 9

_stage_one() {
    local name="$1"
    local src="$_stage_src/$name" dst="$_stage_dst/$name"
    if [ -f "$dst" ] && [ "$(stat -c%s "$dst" 2>/dev/null)" = "$(stat -c%s "$src")" ]; then
        echo "stage_corpus: $name already local ($(stat -c%s "$dst") bytes), skipping"
        return
    fi
    echo "stage_corpus: copying $name ($(du -h "$src" | cut -f1)) to $_stage_dst ..."
    local t0=$(date +%s)
    cp "$src" "$dst.partial" && mv "$dst.partial" "$dst"
    echo "stage_corpus: $name done in $(( $(date +%s) - t0 ))s"
}

_stage_one c4_train.bin
_stage_one c4_val.bin

export TRAIN_PATH="$_stage_dst/c4_train.bin"
export VAL_PATH="$_stage_dst/c4_val.bin"
flock -u 9

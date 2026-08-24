#!/usr/bin/env bash
# Move things out of the way instead of deleting them.
#
# Nothing here is precious enough to keep forever and nothing is cheap enough
# to lose by accident, so retiring is the default and `rm` is the exception.
# Everything lands under $DATA_p330/attic/<date>/ and a register records where
# each thing came from, because a directory of anonymous files is only slightly
# better than no directory at all.
#
#   scripts/retire.sh path [path...]
#   ATTIC=/somewhere/else scripts/retire.sh path
#
# A move within one filesystem is a rename: instant, and costing no space. A
# move across filesystems copies, so retiring twenty gigabytes from local disk
# onto NFS is a real transfer -- check where you are pointing it.
set -euo pipefail

ATTIC="${ATTIC:-${DATA_p330:?DATA_p330 is not set}/attic}"
DEST="$ATTIC/$(date +%Y-%m-%d)"

if [ $# -eq 0 ]; then
    echo "usage: $(basename "$0") <path>..." >&2
    exit 2
fi

mkdir -p "$DEST"
for path in "$@"; do
    if [ ! -e "$path" ]; then
        echo "skip, not there: $path" >&2
        continue
    fi
    target="$DEST/$(basename "$path")"
    n=1
    while [ -e "$target" ]; do
        target="$DEST/$(basename "$path").$n"
        n=$((n + 1))
    done
    mv -- "$path" "$target"
    printf '%s\t%s\t%s\n' "$(date -Is)" "$(readlink -f "$(dirname "$path")")/$(basename "$path")" "$target" \
        >> "$ATTIC/register.tsv"
    echo "retired: $path -> $target"
done

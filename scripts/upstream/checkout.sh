#!/usr/bin/env bash
# Check out a release of openai-agents-python into .vendor/ for reference.
#
# Usage: scripts/upstream/checkout.sh [VERSION]
#   VERSION defaults to the contents of UPSTREAM_VERSION (for example 0.23.1).
#
# The checkout is a read-only reference used while porting; it is ignored by git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="${1:-$(tr -d '[:space:]' < "$ROOT/UPSTREAM_VERSION")}"
VERSION="${VERSION#v}"
REPO="${UPSTREAM_REPO:-https://github.com/openai/openai-agents-python}"
DEST="${UPSTREAM_DIR:-$ROOT/.vendor/openai-agents-python}"
TAG="v$VERSION"

if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$(dirname "$DEST")"
    git init --quiet "$DEST"
    git -C "$DEST" remote add origin "$REPO"
fi

git -C "$DEST" fetch --quiet --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet --force "$TAG"
echo "Checked out openai-agents-python $TAG at $DEST"

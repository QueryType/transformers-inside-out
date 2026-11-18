#!/usr/bin/env bash
# Publish the built site to the gh-pages branch (served by GitHub Pages).
# main stays source-only; gh-pages holds just the built page and its data.
#
#   python site/build_data.py && python site/build_data_more.py   # build first
#   bash site/publish_pages.sh
set -euo pipefail
cd "$(dirname "$0")/.."

for f in site/data/model_data.js site/data/tokenizer.js site/data/more_data.js; do
  [ -f "$f" ] || { echo "missing $f - run site/build_data.py and site/build_data_more.py first"; exit 1; }
done

SRC_COMMIT=$(git rev-parse --short HEAD)
TMP=$(mktemp -d)
trap 'git worktree remove --force "$TMP" 2>/dev/null || true; rm -rf "$TMP"' EXIT

if git ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1; then
  git fetch -q origin gh-pages
  git worktree add -q "$TMP" -B gh-pages origin/gh-pages
else
  git worktree add -q --detach "$TMP"
  git -C "$TMP" checkout -q --orphan gh-pages
fi

git -C "$TMP" rm -rqf --ignore-unmatch . >/dev/null
mkdir -p "$TMP/js" "$TMP/data"
cp site/index.html "$TMP/"
cp site/js/*.js "$TMP/js/"
cp site/data/model_data.js site/data/tokenizer.js site/data/more_data.js "$TMP/data/"
touch "$TMP/.nojekyll"                      # serve files as-is, no Jekyll processing

git -C "$TMP" add -A
if git -C "$TMP" diff --cached --quiet; then
  echo "gh-pages already up to date"
else
  git -C "$TMP" commit -q -m "Publish site from main@$SRC_COMMIT"
  git -C "$TMP" push -q origin gh-pages
  echo "pushed gh-pages (from main@$SRC_COMMIT)"
fi

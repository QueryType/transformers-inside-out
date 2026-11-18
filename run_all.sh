#!/usr/bin/env bash
# Run every lesson in order. Figures go to outputs/.
set -e
cd "$(dirname "$0")"
for f in lessons/[0-9][0-9]_*.py; do
  echo; echo "################ $f ################"
  python "$f"
done

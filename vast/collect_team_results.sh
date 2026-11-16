#!/bin/bash
# Copy every result the team page produced into /workspace/team/<run>/ for download.
# /tmp is wiped when the instance is destroyed - run this first.
mkdir -p /workspace/team
for d in /tmp/autobg_*; do
  mkdir -p /workspace/team/$(basename $d)
  cp $d/*_reflection.jpg $d/*_v2raw.jpg /workspace/team/$(basename $d)/ 2>/dev/null
done
ls -lt /workspace/team

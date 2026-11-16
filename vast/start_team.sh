#!/bin/bash
# Start the team page (upload only, locked settings). Usage: bash vast/start_team.sh PASSWORD
# Login: user autobg + PASSWORD. Open the https://....gradio.live link it prints.
pkill -9 -f reflect_autobg.py 2>/dev/null; sleep 2
cd /workspace/Autobg-Reflections/app
python reflect_autobg.py --full --share --team --password "${1:?give a password}" --port 7870 \
  --lora /workspace/v2.safetensors --backdrop /workspace/floor_satin_light.png

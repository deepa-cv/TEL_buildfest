#!/bin/sh
set -e
MP="${MODEL_PATH:-/app/model_hmda.pkl}"
if [ ! -f "$MP" ]; then
  echo "No model at $MP — training bootstrap HMDA model..."
  export MODEL_PATH="$MP"
  python bootstrap_hmda_model.py
fi
exec python app.py

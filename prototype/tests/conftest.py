import sys
from pathlib import Path

# Allow `from hmda_model import ...` when running pytest from repo root or prototype/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

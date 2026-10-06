"""VDN style adaptation utilities."""
from pathlib import Path
import sys
GPU_ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=GPU_ROOT/'upstream'
if str(UPSTREAM) not in sys.path:sys.path.insert(0,str(UPSTREAM))

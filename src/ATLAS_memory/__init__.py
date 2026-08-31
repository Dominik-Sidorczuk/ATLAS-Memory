# Re-export atlas_memory for case-insensitive / ATLAS_memory import support
from atlas_memory import *
import atlas_memory as _atlas_memory
import sys
sys.modules["ATLAS_memory"] = _atlas_memory

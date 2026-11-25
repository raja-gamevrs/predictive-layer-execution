"""Moved to the Radan engine package (radan_engine.gemma4_decode); this alias keeps the research scripts working."""
import sys

import radan_engine.gemma4_decode as _m

sys.modules[__name__] = _m

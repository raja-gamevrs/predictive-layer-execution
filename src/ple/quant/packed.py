"""Moved to the Radan engine package (radan_engine.quant.packed); this alias keeps the research scripts working."""
import sys

import radan_engine.quant.packed as _m

sys.modules[__name__] = _m

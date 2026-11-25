"""Moved to the Radan engine package (radan_engine.kernels); this alias keeps the research scripts working."""
import sys

import radan_engine.kernels as _m

sys.modules[__name__] = _m

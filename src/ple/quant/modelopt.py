"""Moved to the Radan engine package (radan_engine.quant.modelopt); this alias keeps the research scripts working."""
import sys

import radan_engine.quant.modelopt as _m

sys.modules[__name__] = _m

"""Moved to the Radan engine package (radan_engine.mtp_spec); this alias keeps the research scripts working."""
import sys

import radan_engine.mtp_spec as _m

sys.modules[__name__] = _m

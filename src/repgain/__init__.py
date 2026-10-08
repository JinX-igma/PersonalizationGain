"""repgain: episode-repeatability diagnostics for personalization gains.

Companion package to "How Precise Is a Personalization Gain? Episode
Repeatability in Wearable Activity Recognition". The estimator here answers
one question: if the same fixed personalization adapter is scored on two
different episodes of the same user and condition, how much does the
measured gain move? See `repgain.report.RepeatabilityCheck` for the public
entry point.
"""

from repgain.report import RepeatabilityCheck, RepeatabilityReport

__all__ = ["RepeatabilityCheck", "RepeatabilityReport"]
__version__ = "0.1.0"

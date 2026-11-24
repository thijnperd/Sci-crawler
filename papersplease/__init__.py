"""PapersPlease: an agent-native literature crawler.

The pipeline is deliberately opinionated: search widely, read what you found,
*then* decide whether each paper is actually about the thing the user asked for
before handing back a link. Scope screening is the point of the tool, not a
decoration -- a paper about "muscle" is not a paper about "muscle strength in
older adults".
"""

from .models import Facet, Paper, ScopeSpec, Verdict
from .pipeline import find, read_one, screen_only

__all__ = [
    "Facet",
    "Paper",
    "ScopeSpec",
    "Verdict",
    "__version__",
    "find",
    "read_one",
    "screen_only",
]

__version__ = "0.7.0"

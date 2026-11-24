"""PapersPlease: an agent-native literature crawler.

The pipeline is deliberately opinionated: search widely, read what you found,
*then* decide whether each paper is actually about the thing the user asked for
before handing back a link. Scope screening is the point of the tool, not a
decoration -- a paper about "muscle" is not a paper about "muscle strength in
older adults".
"""

from .models import Paper, ScopeSpec, Facet, Verdict
from .pipeline import find, screen_only, read_one

__all__ = [
    "Paper",
    "ScopeSpec",
    "Facet",
    "Verdict",
    "find",
    "screen_only",
    "read_one",
    "__version__",
]

__version__ = "0.6.0"

"""Golden-file regression harness for dealing_range.compute_structure (Wave 2 item 2A).

compute_structure is one ~500-line pure forward loop whose block ORDER is
load-bearing (the failure-window `continue` pre-empts CHoCH/BOS). It had NO
tests. This package locks its behaviour: real H1 windows are recorded as
(input OHLC -> full output) fixtures committed to the repo, and the CI test
re-runs compute_structure on each and asserts byte-identical output.

No Wave-2 item that touches structure may proceed until this harness is green.
"""

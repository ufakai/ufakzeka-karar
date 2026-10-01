# results

Committed measurements. Every number in a doc, card, report or paper traces to a file in this folder.

Rules, enforced by bench/harness/results.py:

1. Rows are written only from a clean working tree, and each row records the commit of the code that produced it.
2. Files are append-only. A run never overwrites an existing file; a repeated measurement goes to a new file and both stay.
3. A row describes itself: adapter, model, weights revision, the route the answer took, prompt version, host, latency scope.

A file that turns out to be wrong is not deleted or edited; the measurement that replaces it goes to a new file beside it.

Layout: one folder per stage of the project (step0/, step1/, ...).

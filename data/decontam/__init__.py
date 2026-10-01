"""Decontamination of training rows against evaluation test sets (rule 2).

The protocol follows Tülu 3 (arXiv 2411.15124): 8-gram matching on normalised
text, an item counts as overlapping when more than half its tokens sit in
shared 8-grams, and a training set is contaminated when it overlaps more than
2 percent of any evaluation set. MinHash over 5-token shingles at an estimated
Jaccard of 0.8 catches near-duplicates that light edits hide from exact
8-grams.
"""

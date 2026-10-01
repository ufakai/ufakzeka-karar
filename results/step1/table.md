# The instrument

One fine-tuning protocol over four Turkish backbones and five Turkish datasets.
Every cell is a mean and a sample standard
deviation over the seeds listed, built from results/step1/rows.jsonl at 9483ce0f77.

Numbers are after temperature scaling fitted on validation, which is the state the
model would be served in. The raw state is in summary.json. Brier is the sum over
classes, so its range is 0 to 2. Smooth ECE is on the top-label confidence.
The second table for each dataset holds the rest of what the protocol promised: accuracy,
Matthews correlation, the legacy 15-bin ECE, Brier divided by the Brier of always
predicting the class frequencies (below 1 beats that predictor), and the split of
Brier into calibration error and refinement, with each model's parameter count.

## massive_tr

59 classes, 2937 test rows, 5 epochs, seeds [1, 4, 21, 40, 124].

| backbone | rate | macro F1 | Brier | root Brier | smooth ECE | log loss |
|---|---|---|---|---|---|---|
| moganbert | 3e-04 | 0.8465 ± 0.0061 | 0.1936 ± 0.0049 | 0.3111 ± 0.0039 | 0.0198 ± 0.0031 | 0.5219 ± 0.0141 |
| tabibert | 1e-04 | 0.8459 ± 0.0032 | 0.1947 ± 0.0035 | 0.3120 ± 0.0028 | 0.0183 ± 0.0035 | 0.5202 ± 0.0116 |
| berturk | 8e-05 | 0.8428 ± 0.0048 | 0.1979 ± 0.0064 | 0.3145 ± 0.0051 | 0.0258 ± 0.0018 | 0.5339 ± 0.0110 |
| ufakzeka-mlm-1b-bidirectional | 2e-04 | 0.8101 ± 0.0165 | 0.2314 ± 0.0123 | 0.3401 ± 0.0090 | 0.0251 ± 0.0027 | 0.6253 ± 0.0174 |
| ufakzeka | 8e-05 | 0.8100 ± 0.0134 | 0.2193 ± 0.0081 | 0.3311 ± 0.0061 | 0.0270 ± 0.0044 | 0.5898 ± 0.0188 |

| backbone | parameters | accuracy | MCC | ECE, 15 bins | Brier / constant | Brier calibration | Brier refinement |
|---|---|---|---|---|---|---|---|
| moganbert | 149M | 0.8698 ± 0.0031 | 0.8656 ± 0.0031 | 0.0207 ± 0.0037 | 0.1999 ± 0.0050 | -0.0000 ± 0.0001 | 0.1936 ± 0.0049 |
| tabibert | 150M | 0.8699 ± 0.0031 | 0.8657 ± 0.0032 | 0.0189 ± 0.0053 | 0.2011 ± 0.0036 | -0.0001 ± 0.0000 | 0.1947 ± 0.0035 |
| berturk | 111M | 0.8712 ± 0.0046 | 0.8671 ± 0.0048 | 0.0259 ± 0.0025 | 0.2044 ± 0.0066 | -0.0001 ± 0.0003 | 0.1980 ± 0.0062 |
| ufakzeka-mlm-1b-bidirectional | 183M | 0.8421 ± 0.0103 | 0.8371 ± 0.0106 | 0.0256 ± 0.0040 | 0.2391 ± 0.0127 | -0.0003 ± 0.0004 | 0.2317 ± 0.0120 |
| ufakzeka | 183M | 0.8507 ± 0.0062 | 0.8459 ± 0.0065 | 0.0285 ± 0.0045 | 0.2266 ± 0.0084 | -0.0001 ± 0.0003 | 0.2195 ± 0.0081 |

Input budget: berturk 16 tokens (3% of test rows truncated), moganbert 16 tokens (4% of test rows truncated), tabibert 16 tokens (4% of test rows truncated), ufakzeka 14 tokens (4% of test rows truncated), ufakzeka-mlm-1b-bidirectional 14 tokens (4% of test rows truncated).

Reference for the interval below: ufakzeka, macro F1 0.8100.

| against the reference | difference | 95% interval | reading |
|---|---|---|---|
| moganbert | +0.0365 | [+0.0062, +0.0539] | better |
| tabibert | +0.0359 | [+0.0046, +0.0548] | better |
| berturk | +0.0328 | [-0.0002, +0.0517] | unclear |
| ufakzeka+appended_end | +0.0187 | [-0.0097, +0.0334] | unclear |
| ufakzeka-mlm-1b-bidirectional | +0.0001 | [-0.0289, +0.0197] | unclear |

The interval resamples seeds and test examples together, so it covers both
sources of noise, and it is paired, so the shared noise cancels.

## offenseval_tr

2 classes, 3528 test rows, 3 epochs, seeds [1, 4, 21, 40, 124].

| backbone | rate | macro F1 | Brier | root Brier | smooth ECE | log loss |
|---|---|---|---|---|---|---|
| berturk | 2e-05 | 0.8124 ± 0.0025 | 0.1640 ± 0.0007 | 0.2864 ± 0.0006 | 0.0251 ± 0.0018 | 0.2717 ± 0.0011 |
| tabibert | 3e-05 | 0.8098 ± 0.0043 | 0.1677 ± 0.0008 | 0.2896 ± 0.0007 | 0.0208 ± 0.0020 | 0.2742 ± 0.0015 |
| moganbert | 8e-05 | 0.8016 ± 0.0064 | 0.1699 ± 0.0032 | 0.2915 ± 0.0027 | 0.0215 ± 0.0031 | 0.2774 ± 0.0050 |
| ufakzeka | 3e-05 | 0.7994 ± 0.0050 | 0.1739 ± 0.0043 | 0.2949 ± 0.0036 | 0.0210 ± 0.0067 | 0.2860 ± 0.0039 |
| ufakzeka-mlm-1b-bidirectional | 5e-05 | 0.7974 ± 0.0044 | 0.1781 ± 0.0030 | 0.2984 ± 0.0025 | 0.0208 ± 0.0035 | 0.2923 ± 0.0056 |

| backbone | parameters | accuracy | MCC | ECE, 15 bins | Brier / constant | Brier calibration | Brier refinement |
|---|---|---|---|---|---|---|---|
| berturk | 111M | 0.8881 ± 0.0034 | 0.6323 ± 0.0078 | 0.0268 ± 0.0048 | 0.5070 ± 0.0022 | 0.0003 ± 0.0001 | 0.1637 ± 0.0007 |
| tabibert | 149M | 0.8857 ± 0.0023 | 0.6261 ± 0.0062 | 0.0208 ± 0.0022 | 0.5185 ± 0.0023 | 0.0004 ± 0.0001 | 0.1674 ± 0.0007 |
| moganbert | 149M | 0.8821 ± 0.0020 | 0.6118 ± 0.0073 | 0.0220 ± 0.0036 | 0.5253 ± 0.0098 | 0.0002 ± 0.0002 | 0.1697 ± 0.0031 |
| ufakzeka | 182M | 0.8803 ± 0.0061 | 0.6072 ± 0.0124 | 0.0208 ± 0.0072 | 0.5377 ± 0.0132 | 0.0001 ± 0.0002 | 0.1739 ± 0.0041 |
| ufakzeka-mlm-1b-bidirectional | 182M | 0.8783 ± 0.0030 | 0.6013 ± 0.0077 | 0.0213 ± 0.0054 | 0.5505 ± 0.0094 | 0.0001 ± 0.0002 | 0.1779 ± 0.0029 |

Input budget: berturk 64 tokens (6% of test rows truncated), moganbert 67 tokens (6% of test rows truncated), tabibert 70 tokens (5% of test rows truncated), ufakzeka 66 tokens (5% of test rows truncated), ufakzeka-mlm-1b-bidirectional 66 tokens (5% of test rows truncated).

Reference for the interval below: ufakzeka, macro F1 0.7994.

| against the reference | difference | 95% interval | reading |
|---|---|---|---|
| berturk | +0.0130 | [-0.0002, +0.0259] | unclear |
| tabibert | +0.0104 | [-0.0009, +0.0223] | unclear |
| ufakzeka+appended_end | +0.0025 | [-0.0063, +0.0117] | unclear |
| moganbert | +0.0022 | [-0.0100, +0.0150] | unclear |
| ufakzeka-mlm-1b-bidirectional | -0.0020 | [-0.0128, +0.0092] | unclear |

The interval resamples seeds and test examples together, so it covers both
sources of noise, and it is paired, so the shared noise cancels.

## trcola

2 classes, 1000 test rows, 5 epochs, seeds [1, 4, 21, 40, 124].

| backbone | rate | macro F1 | Brier | root Brier | smooth ECE | log loss |
|---|---|---|---|---|---|---|
| moganbert | 1e-04 | 0.6988 ± 0.0102 | 0.3570 ± 0.0074 | 0.4225 ± 0.0044 | 0.0319 ± 0.0058 | 0.5264 ± 0.0088 |
| berturk | 5e-05 | 0.6945 ± 0.0097 | 0.3663 ± 0.0020 | 0.4280 ± 0.0012 | 0.0308 ± 0.0073 | 0.5384 ± 0.0025 |
| tabibert | 3e-05 | 0.6930 ± 0.0056 | 0.3567 ± 0.0025 | 0.4223 ± 0.0015 | 0.0376 ± 0.0128 | 0.5258 ± 0.0037 |
| ufakzeka-mlm-1b-bidirectional | 2e-05 | 0.6308 ± 0.0171 | 0.4035 ± 0.0038 | 0.4491 ± 0.0021 | 0.0314 ± 0.0051 | 0.5852 ± 0.0045 |
| ufakzeka | 1e-05 | 0.5887 ± 0.0412 | 0.4250 ± 0.0109 | 0.4610 ± 0.0059 | 0.0391 ± 0.0113 | 0.6126 ± 0.0141 |

| backbone | parameters | accuracy | MCC | ECE, 15 bins | Brier / constant | Brier calibration | Brier refinement |
|---|---|---|---|---|---|---|---|
| moganbert | 149M | 0.7206 ± 0.0150 | 0.4001 ± 0.0210 | 0.0398 ± 0.0049 | 0.7633 ± 0.0158 | -0.0002 ± 0.0005 | 0.3572 ± 0.0077 |
| berturk | 111M | 0.7144 ± 0.0095 | 0.3894 ± 0.0194 | 0.0369 ± 0.0114 | 0.7831 ± 0.0043 | -0.0004 ± 0.0004 | 0.3667 ± 0.0022 |
| tabibert | 149M | 0.7168 ± 0.0077 | 0.3928 ± 0.0134 | 0.0440 ± 0.0125 | 0.7625 ± 0.0054 | -0.0002 ± 0.0005 | 0.3568 ± 0.0021 |
| ufakzeka-mlm-1b-bidirectional | 182M | 0.6692 ± 0.0061 | 0.2714 ± 0.0253 | 0.0381 ± 0.0039 | 0.8626 ± 0.0080 | 0.0002 ± 0.0007 | 0.4032 ± 0.0036 |
| ufakzeka | 182M | 0.6584 ± 0.0069 | 0.2227 ± 0.0380 | 0.0457 ± 0.0120 | 0.9087 ± 0.0234 | 0.0016 ± 0.0014 | 0.4235 ± 0.0106 |

Input budget: berturk 31 tokens (4% of test rows truncated), moganbert 31 tokens (4% of test rows truncated), tabibert 31 tokens (4% of test rows truncated), ufakzeka 29 tokens (5% of test rows truncated), ufakzeka-mlm-1b-bidirectional 29 tokens (5% of test rows truncated).

Reference for the interval below: ufakzeka, macro F1 0.5887.

| against the reference | difference | 95% interval | reading |
|---|---|---|---|
| moganbert | +0.1100 | [+0.0707, +0.1482] | better |
| berturk | +0.1057 | [+0.0690, +0.1419] | better |
| tabibert | +0.1042 | [+0.0629, +0.1480] | better |
| ufakzeka-mlm-1b-bidirectional | +0.0421 | [-0.0036, +0.0853] | unclear |
| ufakzeka+appended_end | +0.0323 | [-0.0051, +0.0677] | unclear |

The interval resamples seeds and test examples together, so it covers both
sources of noise, and it is paired, so the shared noise cancels.

## mide22

3 classes, 1014 test rows, 10 epochs, seeds [1, 4, 21, 40, 124].

| backbone | rate | macro F1 | Brier | root Brier | smooth ECE | log loss |
|---|---|---|---|---|---|---|
| berturk | 8e-05 | 0.8125 ± 0.0069 | 0.2653 ± 0.0063 | 0.3642 ± 0.0044 | 0.0361 ± 0.0064 | 0.4975 ± 0.0116 |
| moganbert | 1e-04 | 0.8014 ± 0.0044 | 0.2658 ± 0.0038 | 0.3646 ± 0.0026 | 0.0229 ± 0.0030 | 0.4839 ± 0.0084 |
| ufakzeka-mlm-1b-bidirectional | 8e-05 | 0.7953 ± 0.0109 | 0.2807 ± 0.0059 | 0.3746 ± 0.0039 | 0.0253 ± 0.0076 | 0.5114 ± 0.0075 |
| tabibert | 1e-04 | 0.7951 ± 0.0066 | 0.2742 ± 0.0079 | 0.3702 ± 0.0053 | 0.0272 ± 0.0027 | 0.4920 ± 0.0130 |
| ufakzeka | 5e-05 | 0.7745 ± 0.0276 | 0.2955 ± 0.0247 | 0.3841 ± 0.0159 | 0.0349 ± 0.0109 | 0.5338 ± 0.0314 |

| backbone | parameters | accuracy | MCC | ECE, 15 bins | Brier / constant | Brier calibration | Brier refinement |
|---|---|---|---|---|---|---|---|
| berturk | 111M | 0.8298 ± 0.0082 | 0.7129 ± 0.0124 | 0.0402 ± 0.0062 | 0.4503 ± 0.0108 | 0.0003 ± 0.0010 | 0.2650 ± 0.0060 |
| moganbert | 149M | 0.8193 ± 0.0033 | 0.6921 ± 0.0055 | 0.0284 ± 0.0077 | 0.4512 ± 0.0064 | -0.0003 ± 0.0001 | 0.2661 ± 0.0038 |
| ufakzeka-mlm-1b-bidirectional | 182M | 0.8095 ± 0.0075 | 0.6749 ± 0.0117 | 0.0276 ± 0.0092 | 0.4765 ± 0.0100 | -0.0005 ± 0.0002 | 0.2812 ± 0.0058 |
| tabibert | 149M | 0.8105 ± 0.0041 | 0.6803 ± 0.0080 | 0.0335 ± 0.0080 | 0.4654 ± 0.0134 | -0.0003 ± 0.0002 | 0.2744 ± 0.0077 |
| ufakzeka | 182M | 0.7921 ± 0.0296 | 0.6472 ± 0.0445 | 0.0406 ± 0.0135 | 0.5016 ± 0.0420 | -0.0005 ± 0.0005 | 0.2960 ± 0.0251 |

Input budget: berturk 91 tokens (4% of test rows truncated), moganbert 95 tokens (4% of test rows truncated), tabibert 96 tokens (5% of test rows truncated), ufakzeka 90 tokens (5% of test rows truncated), ufakzeka-mlm-1b-bidirectional 90 tokens (5% of test rows truncated).

Reference for the interval below: ufakzeka, macro F1 0.7745.

| against the reference | difference | 95% interval | reading |
|---|---|---|---|
| berturk | +0.0379 | [+0.0094, +0.0692] | better |
| moganbert | +0.0268 | [+0.0006, +0.0546] | better |
| ufakzeka-mlm-1b-bidirectional | +0.0208 | [-0.0082, +0.0477] | unclear |
| tabibert | +0.0206 | [-0.0078, +0.0494] | unclear |
| ufakzeka+appended_end | +0.0130 | [-0.0155, +0.0444] | unclear |

The interval resamples seeds and test examples together, so it covers both
sources of noise, and it is paired, so the shared noise cancels.

## legal_nli_tr

3 classes, 3000 test rows, 3 epochs, seeds [1, 4, 21, 40, 124].

| backbone | rate | macro F1 | Brier | root Brier | smooth ECE | log loss |
|---|---|---|---|---|---|---|
| tabibert | 3e-05 | 0.9975 ± 0.0005 | 0.0052 ± 0.0007 | 0.0509 ± 0.0032 | 0.0032 ± 0.0008 | 0.0166 ± 0.0006 |
| ufakzeka | 1e-04 | 0.9972 ± 0.0011 | 0.0058 ± 0.0014 | 0.0534 ± 0.0065 | 0.0031 ± 0.0007 | 0.0154 ± 0.0025 |
| moganbert | 2e-04 | 0.9971 ± 0.0028 | 0.0064 ± 0.0058 | 0.0532 ± 0.0217 | 0.0037 ± 0.0012 | 0.0170 ± 0.0143 |
| berturk | 8e-05 | 0.9965 ± 0.0008 | 0.0079 ± 0.0019 | 0.0625 ± 0.0075 | 0.0050 ± 0.0006 | 0.0214 ± 0.0040 |

| backbone | parameters | accuracy | MCC | ECE, 15 bins | Brier / constant | Brier calibration | Brier refinement |
|---|---|---|---|---|---|---|---|
| tabibert | 149M | 0.9974 ± 0.0006 | 0.9958 ± 0.0010 | 0.0026 ± 0.0009 | 0.0084 ± 0.0011 | 0.0001 ± 0.0001 | 0.0051 ± 0.0007 |
| ufakzeka | 182M | 0.9965 ± 0.0013 | 0.9944 ± 0.0022 | 0.0019 ± 0.0004 | 0.0093 ± 0.0022 | -0.0000 ± 0.0000 | 0.0058 ± 0.0014 |
| moganbert | 149M | 0.9964 ± 0.0033 | 0.9942 ± 0.0052 | 0.0025 ± 0.0009 | 0.0103 ± 0.0094 | 0.0000 ± 0.0000 | 0.0064 ± 0.0058 |
| berturk | 111M | 0.9956 ± 0.0010 | 0.9929 ± 0.0016 | 0.0037 ± 0.0005 | 0.0127 ± 0.0030 | -0.0001 ± 0.0001 | 0.0080 ± 0.0019 |

Input budget: berturk 512 tokens (58% of test rows truncated), moganbert 512 tokens (46% of test rows truncated), tabibert 512 tokens (61% of test rows truncated), ufakzeka 512 tokens (60% of test rows truncated).

Reference for the interval below: ufakzeka, macro F1 0.9972.

| against the reference | difference | 95% interval | reading |
|---|---|---|---|
| tabibert | +0.0003 | [-0.0013, +0.0018] | tie |
| moganbert | -0.0001 | [-0.0026, +0.0021] | tie |
| berturk | -0.0007 | [-0.0026, +0.0013] | tie |

The interval resamples seeds and test examples together, so it covers both
sources of noise, and it is paired, so the shared noise cancels.

## The gate

One definition. The mean runs over the datasets that can rank a model: massive_tr, offenseval_tr, trcola, mide22.
Each system is read with the pooling its validation score prefers, never its test
score, and every system is listed. A converted backbone passes when it beats
`ufakzeka` here by 1.0 macro F1; the encoders are the numbers it is reported beside.

| system | mean macro F1 | massive_tr | offenseval_tr | trcola | mide22 |
|---|---|---|---|---|---|
| berturk | 0.7905 | 0.8428 | 0.8124 | 0.6945 | 0.8125 |
| moganbert | 0.7871 | 0.8465 | 0.8016 | 0.6988 | 0.8014 |
| tabibert | 0.7859 | 0.8459 | 0.8098 | 0.6930 | 0.7951 |
| ufakzeka | 0.7598 | 0.8287 (appended_end) | 0.8019 (appended_end) | 0.6210 (appended_end) | 0.7876 (appended_end) |
| ufakzeka-mlm-1b-bidirectional | 0.7578 | 0.8115 (mean) | 0.8034 (mean) | 0.6212 (mean) | 0.7952 (mean) |

Poolings were compared on: massive_tr, mide22, offenseval_tr, trcola.
Everywhere else a system has only its stock pooling, so its number there may still
be understating it.

## The gate number, as first defined

The unweighted mean over the five datasets of each backbone's mean macro F1, after
calibration. This is the number the gate reads.

| backbone | mean macro F1 | complete |
|---|---|---|
| berturk | 0.8317 | yes |
| moganbert | 0.8291 | yes |
| tabibert | 0.8283 | yes |
| ufakzeka | 0.7940 | yes |

### The same number without the datasets that cannot rank anything

On legal_nli_tr the whole spread between the best and worst backbone is smaller
than the spread one backbone shows across its own seeds. Re-running the same
model moves the score more than changing the model does, so the dataset adds a
near-constant to every backbone and dilutes the differences the other datasets
found. The gate above is the one first defined and is not changed. This is the
same average with those datasets left out, so the dilution is visible rather
than argued about.

| backbone | mean macro F1 | without | difference |
|---|---|---|---|
| berturk | 0.8317 | 0.7905 | -0.0412 |
| moganbert | 0.8291 | 0.7871 | -0.0420 |
| tabibert | 0.8283 | 0.7859 | -0.0423 |
| ufakzeka | 0.7940 | 0.7432 | -0.0508 |

## What one fixed learning rate costs, on massive_tr

Published Turkish encoder comparisons commonly train every model at the same
learning rate. These are the same backbones and seeds at a fixed 3e-05,
against the per-backbone rates the sweep selected, with the same paired bootstrap.

| backbone | tuned rate | tuned | fixed | difference | 95% interval |
|---|---|---|---|---|---|
| moganbert | 3e-04 | 0.8465 | 0.8159 | -0.0306 | [-0.0459, -0.0119] |
| ufakzeka | 8e-05 | 0.8100 | 0.7848 | -0.0252 | [-0.0419, -0.0038] |
| tabibert | 1e-04 | 0.8459 | 0.8228 | -0.0231 | [-0.0395, -0.0046] |
| berturk | 8e-05 | 0.8428 | 0.8273 | -0.0154 | [-0.0296, +0.0076] |

Ranked by the tuned protocol: moganbert > tabibert > berturk > ufakzeka.
Ranked by the fixed protocol: berturk > tabibert > moganbert > ufakzeka.

The two protocols disagree about which backbone is best. The choice of a
single shared learning rate is not a neutral simplification here: it changes
the answer, so a comparison that fixes one is reporting a property of the rate
as much as a property of the model.

## Pooling ablation, causal backbone

| dataset | pooling | macro F1 | Brier |
|---|---|---|---|
| massive_tr | appended_end | 0.8287 ± 0.0166 | 0.2217 ± 0.0048 |
| massive_tr | mean | 0.7810 ± 0.0123 | 0.2642 ± 0.0053 |
| massive_tr | mean | 0.8115 ± 0.0168 | 0.2392 ± 0.0109 |
| mide22 | appended_end | 0.7876 ± 0.0131 | 0.2825 ± 0.0152 |
| mide22 | mean | 0.7952 ± 0.0092 | 0.2679 ± 0.0114 |
| offenseval_tr | appended_end | 0.8019 ± 0.0073 | 0.1705 ± 0.0024 |
| offenseval_tr | mean | 0.8034 ± 0.0042 | 0.1724 ± 0.0050 |
| trcola | appended_end | 0.6210 ± 0.0239 | 0.4155 ± 0.0068 |
| trcola | mean | 0.6212 ± 0.0244 | 0.4029 ± 0.0034 |

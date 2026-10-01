# Open-weight decision models on one Modal GPU

Rows for four open decision models on HakemBench v1.0, both halves and every probe file, each
run through its own project's published inference code on one NVIDIA L4. No board metric is
computed here.

Produced by bench/baselines/modal_gpu.py with the adapters bench/adapters/decider.py, kev.py and
simple_jev.py (shared plumbing in bench/adapters/systemone.py, tests in
bench/tests/test_step9_adapters.py). The full-run rows record git commit 0c257e2.

## Models

| name in the file names | weights | weights revision | weights licence | code, pinned commit |
|---|---|---|---|---|
| decider-2b | Mapika/decider-2b (decider-2b v11, on Qwen3.5-2B-Base) | 533964dae8be954c5b5e19fa4948e48408094c1e | Apache-2.0 | github.com/Mapika/decider a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f (decider-ai 1.5.0) |
| kev-4b | jaredpalmer/kev-4b (LoRA and pointer head on Qwen/Qwen3.5-4B-Base 1001bb4d826a52d1f399e183466143f4da7b741b) | 139fdd94f1b6a6ad80cc15e08fcb99cac885a101 | Apache-2.0 (base Apache-2.0) | github.com/jaredpalmer/kev 5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109 |
| kev-9b | jaredpalmer/kev-9b (LoRA and pointer head on Qwen/Qwen3.5-9B-Base 68c46c4b3498877f3ef123c856ecfde50c39f404) | 2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c | Apache-2.0 (base Apache-2.0) | github.com/jaredpalmer/kev 5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109 |
| simple-jev-qwen3.5-4b | Qwen/Qwen3.5-4B, read by simple-jev | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | Apache-2.0 | github.com/featherless-ai/simple-jev dae340e30b2e2a2b27dfa1678a7ec5d66e28feca |

All three code repositories are Apache-2.0. Each row's `path_used` carries the route, the code
commit, the settings as loaded, the GPU, the dtype and the package versions; `model_revision`
carries the weights revision.

### Route per model

- decider-2b: `decider.infer.Decider(snapshot).system_one(state, questions)`, CUDA, bf16, its CUDA
  graph engine, plain state-first layout, isolated score levels, every question in its own row,
  all as its decider_config.json sets. Its own temperatures, as shipped: choice 1.164, noul
  1.624, score 1.124. torch 2.14.0, transformers 5.17.0, flash-linear-attention 0.5.2, triton
  3.8.0, numpy 1.26.4, Python 3.12.
- kev-4b: Kev's own Modal endpoint path (skills/kev-deploy/scripts/kev_serve.py): `Checkpoint(run)
  .load("cuda", LoadOptions(dtype=bfloat16, cuda_graphs=True, fused=True))`, `kev.serve.Server`,
  the same three warm-up requests, then `Server.answer` per item, which is what its POST
  /v1/systemone returns. The checkpoint's own temperature as shipped: 2.4061. torch 2.8.0,
  transformers 5.17.0, peft 0.21.0, flash-linear-attention 0.5.2, triton 3.8.0, numpy 2.5.3,
  Python 3.13.
- kev-9b: the same, with CUDA graphs and fused kernels off (see Deviations). Temperature 2.2974.
- simple-jev-qwen3.5-4b: `hf_server.load_service("Qwen/Qwen3.5-4B", revision=..., device="cuda",
  dtype="bfloat16")` with every other setting at the server's defaults, then
  `DecisionService.classify(request)`, which is what its POST /v1/classifier and /v1/systemone
  return. The loader picked the prompt format `shared_examples_binary` for this architecture and
  size (mode "architecture-size"); the rows record it as prompt_version "simple-jev
  shared_examples_binary". No calibration: simple-jev ships none. torch 2.14.0, transformers
  5.17.0, accelerate 1.15.0, flash-linear-attention 0.5.2, triton 3.8.0, numpy 2.5.3, Python 3.13.

Each item goes to the model as the benchmark holds it: its Turkish (or, for the english probe,
English) state and its questions, nothing added, no few-shot, no translation, one call per item,
no retry. The only change to the wire body is that a noul with no criteria leaves the field out
instead of sending null. Probabilities, choice, score and confidence are kept as each model
returns them (decider and Kev round to four places in their own code; simple-jev returns float32
values). A score that passes the top level by float rounding (at most 1e-6) is clipped to it. All
three return their own confidence on choice and score answers, so those rows carry
confidence_source "model": for decider and Kev it is the typed-decision contract's rescaled
maximum ((n p_max - 1)/(n - 1) for a choice, the distance-based form for a score); for
simple-jev it is the largest label probability. Neither is a probability of being right.

## Commands

    # smoke, 20 items of the public half, rows kept off results/
    modal run bench/baselines/modal_gpu.py --model <name> --smoke --files public
    # every file; rows copied to results/step9/runs/ and results/private/step9/runs/
    modal run bench/baselines/modal_gpu.py --model <name>

run with `uv run --with modal==1.5.5` from the repo root, GPU L4 (`STEP9_GPU`, default L4). The
item files go to the project volume `ufakzeka-karar` under step9/items/; the private items went
to that volume only. Every function has a 3-hour timeout, retries 0 and at most one container.

## Rows

Every model answered every question of every file; each file below has the same row count for
all four models, one row per question, one run id per file, no question twice. slots.private.jsonl
is empty and has no rows.

| file | items | rows a model | where |
|---|---|---|---|
| public | 1,198 | 2,166 | `results/step9/runs/<model>-public.jsonl` |
| english-public | 68 | 68 | results/step9/runs/ |
| paraphrase-public | 66 | 66 | results/step9/runs/ |
| permutations-public | 5,190 | 5,190 | results/step9/runs/ |
| slots-public | 423 | 423 | results/step9/runs/ |
| private | 1,722 | 3,069 | `results/private/step9/runs/<model>-private.jsonl` |
| english-private | 91 | 91 | results/private/step9/runs/ |
| paraphrase-private | 134 | 134 | results/private/step9/runs/ |
| permutations-private | 5,610 | 5,610 | results/private/step9/runs/ |
| all | 14,502 | 16,817 | |

Unanswered: 0 for every model and file. No model's own code refused an item (the largest items
have 10 options, 4 levels and about 1,500 characters of state, inside every model's documented
limits), so no .unanswered.jsonl sidecar exists and nothing was truncated.

## GPU time

From results/step9/gpu_runs.jsonl (seconds inside the function, model load included). All runs on NVIDIA L4.

| run | function seconds |
|---|---|
| decider-2b, full | 1,055 |
| kev-4b, full | 1,391 |
| kev-9b, full | 2,673 |
| simple-jev-qwen3.5-4b, full | 7,176 |
| six smoke runs and one crashed start | 701 in the five smokes that finished |

Throughput on the L4 over all files
(items over the summed file seconds): decider-2b 14.1 items a second, kev-4b 12.1, kev-9b 5.7,
simple-jev 2.0.

The crashed start was the first launch: the module could not import in the container, and the
container restarted until the app was stopped by hand (commit 461de45 fixed it). The first
kev-9b smoke ran out of memory (see Deviations) and wrote no summary line.

The weights of the four models are cached on the project volume under hf/ (about 38 GB); they
can be removed with `modal volume rm -r ufakzeka-karar hf/hub/models--<org>--<repo>` if no rerun
is planned.

## Smoke runs

Twenty public items per model before any full run, on the same L4 (results/step9/gpu_runs.jsonl,
the rows with "smoke": true; their rows are in results/step9/smoke/, the second smoke for kev-9b
and simple-jev, run with the settings of the full runs). They are not HakemBench rows for the
board: the same items are in the full runs. Every answer passed the harness's checks
(probabilities summing to 1 within 1e-3, keys equal to the options or levels, the choice among the
most probable).

The first item of a run is slow while kernels compile (decider 65 s, simple-jev 70 s). The
extrapolation used the median latency of the other 19 items, per item: decider 94 ms, kev-4b
119 ms, kev-9b 187 ms, simple-jev 443 ms. Times 14,502 items that is 23, 29, 45 and 107 minutes,
about 3.4 L4 hours.

## Deviations, and why

- kev-9b ran with CUDA graphs and Kev's fused kernels off. With them on, as kev_serve.py loads it,
  it ran out of memory on the 22 GB L4 during warm-up (kev_serve.py lists H100, H200 and L40S for
  the 9B). LoadOptions documents graphs-off and fused-off as the eager path Kev's own reported
  numbers use, and kev.serve exposes both as KEV_CUDA_GRAPHS=0 and KEV_FUSED=0; the weights stay
  bf16 and the answers are equal up to floating-point reassociation. The rows say
  "cuda_graphs False; fused_kernels False"; the route label still reads "as kev_serve.py loads
  it".
- simple-jev: flash-linear-attention 0.5.2 was added to its image. simple-jev's README installs
  only `./hf-server`; without the package transformers runs the Qwen3.5 DeltaNet layers in its
  reference PyTorch code (0.74 items a second in the first smoke). It is
  the kernel transformers asks for and the one decider and Kev install; it changes no prompt and
  no scoring rule. causal-conv1d stayed out, so the short convolution runs in transformers'
  reference code.
- simple-jev's chat model: its README's default chat model for a GPU, a 26B model, does not
  fit a 24 GB GPU in bf16. Qwen/Qwen3.5-4B is the one model of 4B to 9B its README and loader name
  with a tuned prompt format (the "Qwen dense, 4B" reference model), it is multilingual and
  Apache-2.0, so it was used. The README's table gives `strict_mix_repeat2` for it; the loader at
  the pinned commit (hf-server/hf_prompt_policies.py, KNOWN_PROFILES) maps the same configuration
  to `shared_examples_binary`, and the loader's choice is what ran.
- decider-4b exists (Mapika/decider-4b, Apache-2.0) and was not run; the approved row names the
  2B.
- Kev's published evaluation numbers use fp32; its serving path, used here, is bf16 (its README
  says the two differ by at most about 0.03 in probability).

## Things that looked odd

- decider-2b put noul above 0.5 on 60 of the 882 public yes/no questions (6.8%); 394 of those
  882 have gold true. Its README says it is trained on English only.
- simple-jev returned exactly 0.5 on 9 public yes/no questions (a tie between its yes and no
  label logits in bf16).
- decider-2b gives choice or score confidence 0 on 89 public questions, kev-4b on 53, kev-9b on
  7: the rescaled confidence is 0 whenever the distribution is flat enough, by its definition.

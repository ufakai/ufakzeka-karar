# ufakzeka-karar

ufakzeka-karar is an open Turkish decision model by ufak AI. It reads a Turkish text and answers
typed questions about it (a choice from a set of options, a level on an ordered scale, or yes
or no) with a probability for each option, temperature-scaled on validation data, and an
expected error usable as a "not sure" signal, in one forward pass on a CPU for up to ten options, without generating
text. The model card says where the scaling does not transfer: the shipped temperature makes
calibration worse on held-out support questions (0.036 to 0.064 for the released model,
which has since trained on those questions), and on HakemBench the model is somewhat
overconfident. HakemBench is the benchmark it is measured on.

This repository holds the code that built the model (data conversion and labelling, the
backbone conversion, the decision head and its training, calibration, the benchmark harness
and scoring) and the result files that the write-up
(https://ufakai.com/research/ufakzeka-karar) quotes.

## What is here

- `schema/`: the typed-decision request and answer format.
- `data/`: the converters and labelling code for the training data; `data/MANIFEST.yaml`
  lists every source with its URL, its licence as shown and its attribution.
- `model/`: the backbone conversion, the decision head, training and the ablations.
- `calib/`: temperature scaling, the abstain map and the selective-automation metrics.
- `bench/`: the evaluation harness, the model adapters, the leaderboard code and the
  HakemBench build; `bench/hakembench/v1.0/open/` holds the open test set: 2,346 items,
  their provenance, the part A and part B split and the probes.
- `release/`: the release loader (`release/model_repo/karar.py`), the export to the Hugging
  Face format and the public build that produced this repository.
- `results/`: the measurements. Result numbers trace to the committed result files in this
  repository; a few figures from internal audits, such as the human audits of the benchmark's
  gold and the style check of the new moderation comments, are reported as recorded.
- `docs/DATA_CARD.md`: the training data, source by source.

 It also leaves out the raw downloads, the model weights (on Hugging
Face), row-level files from web sources whose text is not ours to publish, and the project's
internal plan, spend ledger and working notes. The cost in the write-up is read from the
providers' dashboards.

## Install and test

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run --group instrument pytest
```

The `instrument` group adds torch and the metric libraries; on Linux torch comes from the
CPU index. The tests need no model weights and no network.

## Run the model

The released model is on Hugging Face as `ufakai/ufakzeka-karar`, chosen by a selection rule fixed
in writing beforehand. It is Run 3, the last of three training runs scored on HakemBench, and its
numbers are not blind: Run 2's new training data was aimed at Run 1's errors on the full test
set in guardrails, moderation and customer support, and Run 3 was trained after Run 2's
guardrail results on the full test set were read, under a protocol fixed in writing before any
of its data, code or runs. The selection rule and its result ship with the code
(`bench/r5_select.py`, `results/step9/r5/selection.json`); the selection file records the flag as
it stood when the run was chosen, on the guardrail track, and it was later extended to moderation
and customer support. Result files in the repository use
the internal names r3, r4 and r5 for runs 1 to 3. Every number of the released model comes
after these readings; its guardrail, moderation and customer support numbers carry the flag "shaped by reading the test
results". On HakemBench v1.0's open set it ranks 7th of 16 rows with a composite of 0.660 (95 percent confidence interval 0.642 to 0.677);
with every model scored on the other four tracks only, its composite is 0.678, 6th of 16. It has 182,494,466
parameters (151,037,186 without the embedding matrix), shipped in fp32 with one global
temperature. Its loader is one file, `release/model_repo/karar.py`, which needs only torch, numpy, safetensors, tokenizers and
huggingface_hub (`release/model_repo/requirements.txt`):

```sh
pip install -r release/model_repo/requirements.txt
```

The example is the model card's quick start: a HakemBench open-set item
(spam-9155fdca2238bef7), one of the karar demo's examples, whose gold answers are
`dolandırıcılık` and yes. `from_pretrained` takes the repo id or a local folder that holds
the model files.

```python
import json
import sys

sys.path.insert(0, "release/model_repo")
from karar import Karar

karar = Karar.from_pretrained("ufakai/ufakzeka-karar")  # a local folder or the repo id
result = karar.decide(
    "Acil! Paketiniz adres hatası yüzünden merkezimizde bekliyor. "
    "24 saat içinde güncelleme yapmazsanız iade edilecek. [bağlantı]",
    {
        "mesaj_turu": {
            "type": "choice",
            "instructions": "Bu mesaj ne tür bir mesaj?",
            "criteria": {
                "işlem bildirimi": "Bir şirketin müşterisine yaptığı işlemle ilgili bildirim.",
                "izinli pazarlama": "Müşterisi olunan bir yerden gelen kampanya ya da duyuru.",
                "istenmeyen reklam": "İlişki kurulmamış bir yerden gelen izinsiz reklam.",
                "dolandırıcılık": "Kişiyi para, şifre ya da bilgi vermeye kandırma girişimi.",
                "kişisel yazışma": "İki kişi arasındaki gündelik mesaj.",
            },
        },
        "oltalama": {
            "type": "noul",
            "instructions": "Bu mesaj, kişiyi para, şifre, doğrulama kodu ya da kişisel bilgi "
            "vermeye kandırmaya çalışan bir dolandırıcılık girişimi mi?",
            "criteria": {
                "true": "Evet, mesaj bir dolandırıcılık girişimi.",
                "false": "Hayır, mesaj bir dolandırıcılık girişimi değil.",
            },
        },
    },
)


def rounded(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: rounded(v) for key, v in value.items()}
    return value


print(json.dumps(rounded(result), ensure_ascii=False, indent=2))
```

It prints (run on CPU from the released folder):

```json
{
  "model": "ufakzeka-karar",
  "answers": {
    "mesaj_turu": {
      "type": "choice",
      "choice": "dolandırıcılık",
      "probabilities": {
        "işlem bildirimi": 0.175,
        "izinli pazarlama": 0.037,
        "istenmeyen reklam": 0.016,
        "dolandırıcılık": 0.763,
        "kişisel yazışma": 0.01
      },
      "confidence": 0.748,
      "abstain": 0.252
    },
    "oltalama": {
      "type": "noul",
      "noul": 0.955,
      "abstain": 0.055
    }
  }
}
```

A choice question takes 2 to 255 options, a score question 2 to 10 levels, and a yes-or-no
(`noul`) question needs no options. Each answer
carries the probabilities and an abstain value, the expected error of that answer, so a
caller can send unsure cases to a person. On HakemBench that signal cannot be relied on to
catch guardrail errors; the model card gives coverage and error per track at each threshold.
The file's docstring describes every field.

## Evaluate on HakemBench

HakemBench v1.0 is fully open. It has its own repository,
[github.com/ufakai/hakembench](https://github.com/ufakai/hakembench), with the harness, the
scoring, the open test set and its probes; the data is also on Hugging Face as
`ufakai/HakemBench`. Its README explains how to run a model on the open set and score it.
There is no submission service. The leaderboard lists only models we ran ourselves,
and anyone can run the open set with the harness and publish their own numbers. The
karar adapter that runs this model on it is `bench/adapters/karar.py` here.

## Licences

The code is under the Apache License 2.0 (`LICENSE`), and so are the weights. Every training
and benchmark source has its own licence, listed in `data/MANIFEST.yaml`; the HakemBench
dataset is under CC BY 4.0, and its repository's `ATTRIBUTION.md` gives the source and
licence of every item of the open set.

## Citation

The papers describe the method and the results; the model and dataset entries
cite the released artefacts.

The papers:

```bibtex
@misc{teke2026ufakzekakarar,
  title         = {ufakzeka-karar: An Open {Turkish} Typed-Decision Model with Order-Invariant Option Scoring},
  author        = {Teke, Sait Furkan},
  year          = {2026},
  howpublished  = {Technical report, ufak AI},
  url           = {https://ufakai.com/reports/ufakzeka-karar.pdf}
}

@misc{teke2026hakembench,
  title         = {{HakemBench}: A {Turkish} Benchmark of Typed Decisions},
  author        = {Teke, Sait Furkan},
  year          = {2026},
  howpublished  = {Technical report, ufak AI},
  url           = {https://ufakai.com/reports/hakembench.pdf}
}
```

The model and the dataset:

```bibtex
@misc{ufakzekakarar2026,
  title         = {ufakzeka-karar},
  author        = {{ufak AI}},
  year          = {2026},
  howpublished  = {\url{https://huggingface.co/ufakai/ufakzeka-karar}}
}

@misc{hakembench2026,
  title         = {{HakemBench}},
  author        = {{ufak AI}},
  year          = {2026},
  version       = {1.0},
  howpublished  = {\url{https://huggingface.co/datasets/ufakai/HakemBench}}
}
```

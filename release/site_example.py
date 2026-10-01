"""The karar page's request example, answered by the release model folder.

    uv run --group local python -m release.site_example <model folder>

The request is karar.py's docstring example with a yes/no and a score question added. The
answer is written unchanged to results/step9/site_example.json with the sha256 of the weights
it came from, so the page shows the release model's own output, never a typed-in one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("results/step9/site_example.json")
REQUEST = {
    "model": "ufakzeka-karar",
    "state": "Faturam iki kez kesildi, iade istiyorum.",
    "questions": {
        "konu": {"type": "choice", "instructions": "Mesajın konusu nedir?",
                 "criteria": {"fatura": None, "kargo": None, "iade": None}},
        "temsilci": {"type": "noul",
                     "instructions": "Bu mesaj bir müşteri temsilcisine yönlendirilmeli mi?"},
        "ofke": {"type": "score", "instructions": "Müşteri ne kadar öfkeli?",
                 "criteria": ["Sakin", "Rahatsız", "Öfkeli"]},
    },
}  # fmt: skip


def main(folder: str) -> None:
    root = Path(folder).expanduser()
    sys.path.insert(0, str(root))
    from karar import Karar

    karar = Karar.from_pretrained(root)
    calibrator = json.loads((root / "calibrator.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    response = karar.decide(REQUEST["state"], REQUEST["questions"])
    out = {"source_run": config["source_run"], "weights_sha256": calibrator["weights_sha256"],
           "request": REQUEST, "response": response}  # fmt: skip
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(response, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

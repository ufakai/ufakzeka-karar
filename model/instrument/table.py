"""The step 1 results table, built from committed rows.

Reads results/step1/rows.jsonl and comparisons.json and writes table.md and
summary.json. It computes nothing a run did not already measure: every cell
is a mean and a sample standard deviation over the seeds of one committed
row set, so the table can be rebuilt from the repository alone.

The gate number is the unweighted mean over datasets of a backbone's
mean macro-F1. It is unweighted so a large dataset cannot decide a backbone
comparison on its own.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from model.instrument.plan import BACKBONES, DATASETS, system_name

# Reported in this order, with where each is read from in a row.
METRICS = (
    ("macro F1", "macro_f1", 4),
    ("Brier", "brier", 4),
    ("root Brier", "root_brier", 4),
    ("smooth ECE", "smooth_ece", 4),
    ("log loss", "logloss", 4),
)


# What the protocol also promised for every backbone and dataset. Each row has always
# carried them; they never reached the summary or the table, so the promise was
# kept in the data and broken in the report.
MORE_METRICS = (
    ("accuracy", "accuracy", 4),
    ("MCC", "mcc", 4),
    ("ECE, 15 bins", "ece_15", 4),
    ("Brier / constant", "brier_normalised", 4),
    ("Brier calibration", "brier_calibration_error", 4),
    ("Brier refinement", "brier_refinement", 4),
)


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _spread(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def summarise(rows: list[dict]) -> dict:
    """Mean and sample standard deviation per dataset, backbone, pooling and state."""
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        # The attention arm is part of the key, or the causal and bidirectional runs of one backbone
        # would be averaged into a single row and the comparison the arm exists to make would
        # disappear into a mean.
        arm = row.get("attention", "causal")
        grouped.setdefault((row["dataset"], row["backbone"], row["pooling"], arm), []).append(row)

    out: dict[str, dict] = {}
    for (dataset, backbone, pooling, arm), group in sorted(grouped.items()):
        entry = {
            "dataset": dataset,
            "backbone": backbone,
            "pooling": pooling,
            "attention": arm,
            "seeds": sorted(row["seed"] for row in group),
            "learning_rate": group[0]["learning_rate"],
            "n_test": group[0]["n_test"],
            "n_classes": group[0]["n_classes"],
            "max_length": group[0]["run"]["max_length"],
            "truncated_test": group[0]["run"]["truncated_share"].get("test"),
            "parameters_total": group[0]["run"]["parameters_total"],
            "selected_steps": sorted(row["selection"]["step"] for row in group),
            # The validation score the checkpoint was chosen on, lower is
            # better. The gate picks a pooling on this and never on test.
            "selection_score": statistics.fmean(row["selection"]["score"] for row in group),
        }
        for state in ("raw", "calibrated"):
            for _, key, _ in METRICS + MORE_METRICS:
                mean, spread = _spread([row[state][key] for row in group])
                entry[f"{state}_{key}"] = mean
                entry[f"{state}_{key}_sd"] = spread
        mean, spread = _spread([row["inverse_temperature"] for row in group])
        entry["inverse_temperature"] = mean
        entry["inverse_temperature_sd"] = spread
        suffix = "" if arm == "causal" else f"/{arm}"
        out[f"{dataset}/{backbone}/{pooling}{suffix}"] = entry
    return out


def saturated_datasets(summary: dict) -> list[str]:
    """Datasets that cannot rank the backbones, by their own numbers.

    A dataset is saturated when the whole spread between the best and worst
    backbone is smaller than the largest spread one backbone shows across its
    own seeds. Re-running the same backbone then moves the score more than
    changing the backbone does, so the dataset has no ranking to give and
    contributes a near-constant to any average taken over datasets.

    This is a computed property of the results, not a judgement about a
    dataset, so it is stated the same way whatever the numbers turn out to be.
    """
    out = []
    for dataset in DATASETS:
        entries = [
            entry
            for entry in summary.values()
            if entry["dataset"] == dataset.key and entry["pooling"] == "stock"
        ]
        if len(entries) < 2:
            continue
        means = [entry["calibrated_macro_f1"] for entry in entries]
        spreads = [entry["calibrated_macro_f1_sd"] for entry in entries]
        if max(means) - min(means) < max(spreads):
            out.append(dataset.key)
    return out


def gate_numbers(summary: dict, exclude: tuple[str, ...] = ()) -> dict[str, dict]:
    """The unweighted mean over datasets of each backbone's mean macro-F1."""
    wanted = {dataset.key for dataset in DATASETS} - set(exclude)
    out: dict[str, dict] = {}
    for backbone in BACKBONES:
        per_dataset = {
            entry["dataset"]: entry["calibrated_macro_f1"]
            for entry in summary.values()
            if entry["backbone"] == backbone.key
            and entry["pooling"] == "stock"
            and entry.get("attention", "causal") == "causal"
            and entry["dataset"] not in exclude
        }
        out[backbone.key] = {
            "mean_macro_f1": statistics.fmean(per_dataset.values()) if per_dataset else None,
            "datasets": dict(sorted(per_dataset.items())),
            "complete": set(per_dataset) == wanted,
            "missing": sorted(wanted - set(per_dataset)),
        }
    return out


def decision_gate(summary: dict) -> dict:
    """The one gate definition, computed rather than described.

    Three earlier entries each amended the gate and the code implemented none
    of them. This is all three at once. The average runs over the datasets
    that can rank anything. Each system is read with the pooling that
    its own validation score prefers, never its test score, so a baseline
    cannot lose the comparison by being pooled badly. And every system
    is listed, each attention arm and each encoder, so the number a converted
    backbone has to beat and the numbers it would like to beat sit in one
    table, with a system that lacks a dataset shown as incomplete instead of
    left out.

    `poolings` says how many poolings were compared on each dataset. Where it
    is 1 the choice was not a choice, which is the honest state of every
    dataset but massive_tr until the ablation is run on the rest.
    """
    saturated = saturated_datasets(summary)
    ranking = [dataset.key for dataset in DATASETS if dataset.key not in saturated]
    systems: dict[str, dict[str, list[dict]]] = {}
    for entry in summary.values():
        if entry["dataset"] not in ranking:
            continue
        name = system_name(entry["backbone"], entry.get("attention", "causal"))
        systems.setdefault(name, {}).setdefault(entry["dataset"], []).append(entry)

    out: dict[str, dict] = {}
    for name, per_dataset in sorted(systems.items()):
        chosen = {}
        for dataset, candidates in per_dataset.items():
            best = min(candidates, key=lambda e: e["selection_score"])
            chosen[dataset] = {
                "macro_f1": best["calibrated_macro_f1"],
                "pooling": best["pooling"],
                "poolings": len(candidates),
            }
        complete = set(chosen) == set(ranking)
        out[name] = {
            "mean_macro_f1": (
                statistics.fmean(item["macro_f1"] for item in chosen.values()) if complete else None
            ),
            "datasets": dict(sorted(chosen.items())),
            "complete": complete,
            "missing": sorted(set(ranking) - set(chosen)),
        }
    return {"ranking_datasets": ranking, "saturated": saturated, "systems": out}


def decision_gate_section(gate: dict) -> list[str]:
    ranking = gate["ranking_datasets"]
    lines = [
        "## The gate",
        "",
        "One definition. The mean runs over the datasets that can rank a model: "
        + ", ".join(ranking)
        + ".",
        "Each system is read with the pooling its validation score prefers, never its test",
        "score, and every system is listed. A converted backbone passes when it beats",
        "`ufakzeka` here by 1.0 macro F1; the encoders are the numbers it is reported beside.",
        "",
        "| system | mean macro F1 | " + " | ".join(ranking) + " |",
        "|---|---|" + "---|" * len(ranking),
    ]
    ordered = sorted(gate["systems"].items(), key=lambda kv: -(kv[1]["mean_macro_f1"] or 0))
    for name, item in ordered:
        mean = "incomplete" if item["mean_macro_f1"] is None else f"{item['mean_macro_f1']:.4f}"
        cells = []
        for dataset in ranking:
            got = item["datasets"].get(dataset)
            if got is None:
                cells.append("missing")
                continue
            note = "" if got["pooling"] == "stock" else f" ({got['pooling']})"
            cells.append(f"{got['macro_f1']:.4f}{note}")
        lines.append(f"| {name} | {mean} | " + " | ".join(cells) + " |")
    compared = sorted(
        {
            dataset
            for item in gate["systems"].values()
            for dataset, got in item["datasets"].items()
            if got["poolings"] > 1
        }
    )
    lines += [
        "",
        "Poolings were compared on: " + (", ".join(compared) if compared else "no dataset") + ".",
        "Everywhere else a system has only its stock pooling, so its number there may still",
        "be understating it.",
        "",
    ]
    return lines


def _cell(entry: dict, key: str, digits: int, state: str) -> str:
    return f"{entry[f'{state}_{key}']:.{digits}f} ± {entry[f'{state}_{key}_sd']:.{digits}f}"


def protocol_section(protocol: dict, summary: dict) -> list[str]:
    """What fixing one learning rate for every backbone costs, and whether it reorders them."""
    lines: list[str] = []
    for dataset, per_backbone in sorted(protocol.items()):
        compared = {
            backbone: item
            for backbone, item in per_backbone.items()
            if item.get("status") == "compared"
        }
        if not compared:
            continue
        tuned = {
            entry["backbone"]: entry["calibrated_macro_f1"]
            for entry in summary.values()
            if entry["dataset"] == dataset and entry["pooling"] == "stock"
        }
        fixed_rate = next(iter(compared.values()))["fixed_rate"]
        rows = []
        for backbone, item in compared.items():
            arm = next(s for s in item["systems"] if s["system"] == "fixed")
            rows.append((backbone, item["tuned_rate"], tuned.get(backbone), arm))
        lines += [
            "",
            f"## What one fixed learning rate costs, on {dataset}",
            "",
            "Published Turkish encoder comparisons commonly train every model at the same",
            f"learning rate. These are the same backbones and seeds at a fixed {fixed_rate:.0e},",
            "against the per-backbone rates the sweep selected, with the same paired bootstrap.",
            "",
            "| backbone | tuned rate | tuned | fixed | difference | 95% interval |",
            "|---|---|---|---|---|---|",
        ]
        for backbone, rate, tuned_f1, arm in sorted(rows, key=lambda r: r[3]["difference"]):
            shown = "n/a" if tuned_f1 is None else f"{tuned_f1:.4f}"
            lines.append(
                f"| {backbone} | {rate:.0e} | {shown} | {arm['mean']:.4f} | "
                f"{arm['difference']:+.4f} | [{arm['low']:+.4f}, {arm['high']:+.4f}] |"
            )
        order_tuned = [b for b, _ in sorted(tuned.items(), key=lambda kv: -kv[1]) if b in compared]
        order_fixed = [r[0] for r in sorted(rows, key=lambda r: -r[3]["mean"])]
        lines += ["", f"Ranked by the tuned protocol: {' > '.join(order_tuned)}."]
        lines.append(f"Ranked by the fixed protocol: {' > '.join(order_fixed)}.")
        if order_tuned != order_fixed:
            lines += [
                "",
                "The two protocols disagree about which backbone is best. The choice of a",
                "single shared learning rate is not a neutral simplification here: it changes",
                "the answer, so a comparison that fixes one is reporting a property of the rate",
                "as much as a property of the model.",
            ]
    return lines


def render(
    summary: dict, gates: dict, comparisons: dict, commit: str, protocol: dict | None = None
) -> str:
    """The committed table. Every number here comes from results/step1/rows.jsonl."""
    lines = [
        "# The instrument",
        "",
        "One fine-tuning protocol over four Turkish backbones and five Turkish datasets.",
        "Every cell is a mean and a sample standard",
        f"deviation over the seeds listed, built from results/step1/rows.jsonl at {commit[:10]}.",
        "",
        "Numbers are after temperature scaling fitted on validation, which is the state the",
        "model would be served in. The raw state is in summary.json. Brier is the sum over",
        "classes, so its range is 0 to 2. Smooth ECE is on the top-label confidence.",
        "The second table for each dataset holds the rest of what the protocol promised: accuracy,",
        "Matthews correlation, the legacy 15-bin ECE, Brier divided by the Brier of always",
        "predicting the class frequencies (below 1 beats that predictor), and the split of",
        "Brier into calibration error and refinement, with each model's parameter count.",
        "",
    ]

    for dataset in DATASETS:
        entries = [
            entry
            for entry in summary.values()
            if entry["dataset"] == dataset.key and entry["pooling"] == "stock"
        ]
        if not entries:
            continue
        first = entries[0]
        lines += [
            f"## {dataset.key}",
            "",
            f"{first['n_classes']} classes, {first['n_test']} test rows, "
            f"{dataset.epochs:g} epochs, seeds {first['seeds']}.",
            "",
            "| backbone | rate | " + " | ".join(name for name, _, _ in METRICS) + " |",
            "|---|---|" + "---|" * len(METRICS),
        ]
        for entry in sorted(entries, key=lambda e: -e["calibrated_macro_f1"]):
            cells = " | ".join(
                _cell(entry, key, digits, "calibrated") for _, key, digits in METRICS
            )
            # The arm belongs in the label. Without it the two arms of one
            # backbone render as two rows under the same name, and the table
            # cannot be read at all.
            name = system_name(entry["backbone"], entry.get("attention", "causal"))
            lines.append(f"| {name} | {entry['learning_rate']:.0e} | {cells} |")
        lines += [
            "",
            "| backbone | parameters | " + " | ".join(name for name, _, _ in MORE_METRICS) + " |",
            "|---|---|" + "---|" * len(MORE_METRICS),
        ]
        for entry in sorted(entries, key=lambda e: -e["calibrated_macro_f1"]):
            cells = " | ".join(
                _cell(entry, key, digits, "calibrated") for _, key, digits in MORE_METRICS
            )
            name = system_name(entry["backbone"], entry.get("attention", "causal"))
            size = entry.get("parameters_total")
            shown = "n/a" if not size else f"{size / 1e6:.0f}M"
            lines.append(f"| {name} | {shown} | {cells} |")
        lengths = {
            system_name(e["backbone"], e.get("attention", "causal")): (
                e["max_length"],
                e["truncated_test"],
            )
            for e in entries
        }
        notes = ", ".join(
            f"{name} {length} tokens ({share:.0%} of test rows truncated)"
            for name, (length, share) in sorted(lengths.items())
            if length is not None and share is not None
        )
        if notes:
            lines += ["", f"Input budget: {notes}.", ""]
        else:
            lines.append("")

        for item in comparisons.get(dataset.key, []):
            if item["difference"] is None:
                lines.append(
                    f"Reference for the interval below: {item['system']}, "
                    f"macro F1 {item['mean']:.4f}."
                )
        rows = [item for item in comparisons.get(dataset.key, []) if item["difference"] is not None]
        if rows:
            lines += [
                "",
                "| against the reference | difference | 95% interval | reading |",
                "|---|---|---|---|",
            ]
            for item in rows:
                reading = "tie" if item["tie"] else ("better" if item["low"] > 0 else "unclear")
                if item["high"] < 0:
                    reading = "worse"
                lines.append(
                    f"| {item['system']} | {item['difference']:+.4f} | "
                    f"[{item['low']:+.4f}, {item['high']:+.4f}] | {reading} |"
                )
            lines += [
                "",
                "The interval resamples seeds and test examples together, so it covers both",
                "sources of noise, and it is paired, so the shared noise cancels.",
                "",
            ]

    lines += decision_gate_section(decision_gate(summary))
    lines += [
        "## The gate number, as first defined",
        "",
        "The unweighted mean over the five datasets of each backbone's mean macro F1, after",
        "calibration. This is the number the gate reads.",
        "",
        "| backbone | mean macro F1 | complete |",
        "|---|---|---|",
    ]
    for key, gate in sorted(gates.items(), key=lambda kv: -(kv[1]["mean_macro_f1"] or 0)):
        value = "not yet" if gate["mean_macro_f1"] is None else f"{gate['mean_macro_f1']:.4f}"
        state = "yes" if gate["complete"] else f"missing {', '.join(gate['missing'])}"
        lines.append(f"| {key} | {value} | {state} |")

    saturated = saturated_datasets(summary)
    if saturated:
        narrowed = gate_numbers(summary, exclude=tuple(saturated))
        names = ", ".join(saturated)
        lines += [
            "",
            "### The same number without the datasets that cannot rank anything",
            "",
            f"On {names} the whole spread between the best and worst backbone is smaller",
            "than the spread one backbone shows across its own seeds. Re-running the same",
            "model moves the score more than changing the model does, so the dataset adds a",
            "near-constant to every backbone and dilutes the differences the other datasets",
            "found. The gate above is the one first defined and is not changed. This is the",
            "same average with those datasets left out, so the dilution is visible rather",
            "than argued about.",
            "",
            "| backbone | mean macro F1 | without | difference |",
            "|---|---|---|---|",
        ]
        for key, gate in sorted(narrowed.items(), key=lambda kv: -(kv[1]["mean_macro_f1"] or 0)):
            full = gates[key]["mean_macro_f1"]
            value = gate["mean_macro_f1"]
            if value is None or full is None:
                continue
            lines.append(f"| {key} | {full:.4f} | {value:.4f} | {value - full:+.4f} |")

    if protocol:
        lines += protocol_section(protocol, summary)

    ablations = [entry for entry in summary.values() if entry["pooling"] != "stock"]
    if ablations:
        lines += [
            "",
            "## Pooling ablation, causal backbone",
            "",
            "| dataset | pooling | macro F1 | Brier |",
            "|---|---|---|---|",
        ]
        for entry in sorted(ablations, key=lambda e: (e["dataset"], e["pooling"])):
            lines.append(
                f"| {entry['dataset']} | {entry['pooling']} | "
                f"{_cell(entry, 'macro_f1', 4, 'calibrated')} | "
                f"{_cell(entry, 'brier', 4, 'calibrated')} |"
            )
    return "\n".join(lines) + "\n"


def build(results_dir: Path, commit: str) -> tuple[Path, Path]:
    rows = load_rows(results_dir / "rows.jsonl")
    summary = summarise(rows)
    gates = gate_numbers(summary)
    comparisons_path = results_dir / "comparisons.json"
    comparisons = (
        json.loads(comparisons_path.read_text(encoding="utf-8"))
        if comparisons_path.is_file()
        else {}
    )
    summary_path = results_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"summary": summary, "gate": gates, "decision_gate": decision_gate(summary)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    protocol_path = results_dir / "fixed_rate.json"
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    )
    table_path = results_dir / "table.md"
    table_path.write_text(render(summary, gates, comparisons, commit, protocol), encoding="utf-8")
    return table_path, summary_path

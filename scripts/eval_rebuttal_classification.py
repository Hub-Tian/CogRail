# #!/usr/bin/env python3
# # Copyright 2026 the LlamaFactory team.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");

# import argparse
# import json
# import re
# from collections import Counter
# from pathlib import Path
# from typing import Any, Optional


# CLASSES = ("A", "B", "C")
# CLASS_NAMES = {
#     "A": "Safe",
#     "B": "Potential Threat",
#     "C": "Serious Threat",
# }


# ANSWER_PATTERNS = (
#     re.compile(r"\banswer\s*[:：]\s*[\(\[]?\s*([ABC])\b", re.IGNORECASE),
#     re.compile(r"^[\s\"'`]*[\(\[]?\s*([ABC])\s*[\.\)：:\]\)]", re.IGNORECASE),
#     re.compile(r"\boption\s*([ABC])\b", re.IGNORECASE),
#     re.compile(r"\b([ABC])\s*[\.\)]\s*(safe|potential|serious)\b", re.IGNORECASE),
# )
# WORD_PATTERNS = (
#     ("C", re.compile(r"\bserious\s+threat\b", re.IGNORECASE)),
#     ("B", re.compile(r"\bpotential\s+threat\b", re.IGNORECASE)),
#     ("A", re.compile(r"\bsafe\b", re.IGNORECASE)),
# )


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Evaluate rebuttal prediction files as A/B/C classification and emit LaTeX tables."
#     )
#     parser.add_argument(
#         "--prediction-root",
#         type=Path,
#         default=Path("/data/Projects/ZhangQY/LLaMA-Factory/saves/rebuttal_MRSI_predictions/Cog-MRSI_RailThreat_masked_Type_I_MRSI_sft_20260606_165800"),
#         help="Directory containing one subdirectory per model.",
#     )
#     parser.add_argument(
#         "--prediction-file",
#         type=str,
#         default="generated_predictions.json",
#         help="Prediction filename under each model directory. JSON list and JSONL are supported.",
#     )
#     parser.add_argument(
#         "--output-prefix",
#         type=Path,
#         default=None,
#         help="Output prefix. Defaults to <prediction-root>/rebuttal_metrics.",
#     )
#     parser.add_argument(
#         "--digits",
#         type=int,
#         default=2,
#         help="Number of decimal places for percentages in LaTeX tables.",
#     )
#     parser.add_argument(
#         "--include-invalid",
#         action="store_true",
#         help="Keep samples whose prediction label cannot be parsed in the denominator.",
#     )
#     return parser.parse_args()


# def load_records(path: Path) -> list[dict[str, Any]]:
#     if path.suffix == ".jsonl":
#         with path.open("r", encoding="utf-8") as f:
#             return [json.loads(line) for line in f if line.strip()]

#     with path.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     if isinstance(data, list):
#         return data
#     raise ValueError(f"Expected a JSON list or JSONL records: {path}")


# def extract_label(text: Any) -> Optional[str]:
#     if text is None:
#         return None

#     value = str(text).strip()
#     if not value:
#         return None

#     for pattern in ANSWER_PATTERNS:
#         match = pattern.search(value)
#         if match:
#             return match.group(1).upper()

#     for label, pattern in WORD_PATTERNS:
#         if pattern.search(value):
#             return label

#     stripped = value.strip().upper().rstrip(".。")
#     if stripped in CLASSES:
#         return stripped

#     return None


# def get_gold_text(record: dict[str, Any]) -> Any:
#     if "label" in record and record["label"]:
#         return record["label"]

#     conversations = record.get("conversations") or []
#     for message in reversed(conversations):
#         if message.get("from") == "gpt":
#             return message.get("value")

#     return None


# def safe_div(numerator: float, denominator: float) -> float:
#     return numerator / denominator if denominator else 0.0


# def compute_metrics(records: list[dict[str, Any]], include_invalid: bool) -> dict[str, Any]:
#     gold_labels: list[str] = []
#     pred_labels: list[Optional[str]] = []
#     invalid_gold = 0
#     invalid_pred = 0

#     for record in records:
#         gold = extract_label(get_gold_text(record))
#         pred = extract_label(record.get("predict"))

#         if gold is None:
#             invalid_gold += 1
#             continue

#         if pred is None:
#             invalid_pred += 1
#             if not include_invalid:
#                 continue

#         gold_labels.append(gold)
#         pred_labels.append(pred)

#     support = Counter(gold_labels)
#     total = len(gold_labels)
#     correct = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred)

#     per_class = {}
#     for label in CLASSES:
#         tp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred == label)
#         fp = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold != label and pred == label)
#         fn = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred != label)
#         precision = safe_div(tp, tp + fp)
#         recall = safe_div(tp, tp + fn)
#         f1 = safe_div(2 * precision * recall, precision + recall)
#         per_class[label] = {
#             "name": CLASS_NAMES[label],
#             "support": support[label],
#             "tp": tp,
#             "fp": fp,
#             "fn": fn,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#         }

#     macro_precision = sum(per_class[label]["precision"] for label in CLASSES) / len(CLASSES)
#     macro_recall = sum(per_class[label]["recall"] for label in CLASSES) / len(CLASSES)
#     macro_f1 = sum(per_class[label]["f1"] for label in CLASSES) / len(CLASSES)
#     weighted_precision = safe_div(sum(per_class[label]["precision"] * support[label] for label in CLASSES), total)
#     weighted_recall = safe_div(sum(per_class[label]["recall"] * support[label] for label in CLASSES), total)
#     weighted_f1 = safe_div(sum(per_class[label]["f1"] * support[label] for label in CLASSES), total)

#     return {
#         "samples": total,
#         "raw_samples": len(records),
#         "invalid_gold": invalid_gold,
#         "invalid_pred": invalid_pred,
#         "accuracy": safe_div(correct, total),
#         "macro_precision": macro_precision,
#         "macro_recall": macro_recall,
#         "macro_f1": macro_f1,
#         "weighted_precision": weighted_precision,
#         "weighted_recall": weighted_recall,
#         "weighted_f1": weighted_f1,
#         "per_class": per_class,
#     }


# def discover_prediction_files(root: Path, prediction_file: str) -> list[Path]:
#     direct_file = root / prediction_file
#     if direct_file.is_file():
#         return [direct_file]

#     return sorted(path for path in root.glob(f"*/{prediction_file}") if path.is_file())


# def pct(value: float, digits: int) -> str:
#     return f"{value * 100:.{digits}f}"


# def escape_latex(value: str) -> str:
#     replacements = {
#         "\\": r"\textbackslash{}",
#         "&": r"\&",
#         "%": r"\%",
#         "$": r"\$",
#         "#": r"\#",
#         "_": r"\_",
#         "{": r"\{",
#         "}": r"\}",
#         "~": r"\textasciitilde{}",
#         "^": r"\textasciicircum{}",
#     }
#     return "".join(replacements.get(char, char) for char in value)


# def format_summary_table(results: dict[str, dict[str, Any]], digits: int) -> str:
#     lines = [
#         r"\begin{tabular}{lrrrrrrr}",
#         r"\toprule",
#         r"Model & N & Invalid & Accuracy & Macro-P & Macro-R & Macro-F1 & Weighted-F1 \\",
#         r"\midrule",
#     ]
#     for model, metrics in results.items():
#         lines.append(
#             " & ".join(
#                 [
#                     escape_latex(model),
#                     str(metrics["samples"]),
#                     str(metrics["invalid_pred"]),
#                     pct(metrics["accuracy"], digits),
#                     pct(metrics["macro_precision"], digits),
#                     pct(metrics["macro_recall"], digits),
#                     pct(metrics["macro_f1"], digits),
#                     pct(metrics["weighted_f1"], digits),
#                 ]
#             )
#             + r" \\"
#         )

#     lines.extend([r"\bottomrule", r"\end{tabular}"])
#     return "\n".join(lines) + "\n"


# def format_per_class_table(results: dict[str, dict[str, Any]], digits: int) -> str:
#     lines = [
#         r"\begin{tabular}{llrrrr}",
#         r"\toprule",
#         r"Model & Class & Support & Precision & Recall & F1 \\",
#         r"\midrule",
#     ]
#     for model, metrics in results.items():
#         for label in CLASSES:
#             class_metrics = metrics["per_class"][label]
#             lines.append(
#                 " & ".join(
#                     [
#                         escape_latex(model),
#                         f"{label} ({escape_latex(class_metrics['name'])})",
#                         str(class_metrics["support"]),
#                         pct(class_metrics["precision"], digits),
#                         pct(class_metrics["recall"], digits),
#                         pct(class_metrics["f1"], digits),
#                     ]
#                 )
#                 + r" \\"
#             )

#     lines.extend([r"\bottomrule", r"\end{tabular}"])
#     return "\n".join(lines) + "\n"


# def main() -> None:
#     args = parse_args()
#     prediction_files = discover_prediction_files(args.prediction_root, args.prediction_file)
#     if not prediction_files:
#         raise SystemExit(f"No prediction files found under {args.prediction_root} with name {args.prediction_file}.")

#     results = {}
#     for path in prediction_files:
#         model = path.parent.name if path.parent != args.prediction_root else path.stem
#         records = load_records(path)
#         results[model] = compute_metrics(records, include_invalid=args.include_invalid)

#     output_prefix = args.output_prefix or args.prediction_root / "rebuttal_metrics"
#     output_prefix.parent.mkdir(parents=True, exist_ok=True)

#     summary_table = format_summary_table(results, args.digits)
#     per_class_table = format_per_class_table(results, args.digits)

#     json_path = output_prefix.with_suffix(".json")
#     summary_tex_path = output_prefix.with_suffix(".tex")
#     per_class_tex_path = output_prefix.with_name(output_prefix.name + "_per_class").with_suffix(".tex")

#     with json_path.open("w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     summary_tex_path.write_text(summary_table, encoding="utf-8")
#     per_class_tex_path.write_text(per_class_table, encoding="utf-8")

#     print(summary_table)
#     print(f"Saved metrics JSON to {json_path}")
#     print(f"Saved summary LaTeX table to {summary_tex_path}")
#     print(f"Saved per-class LaTeX table to {per_class_tex_path}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0.

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Optional


CLASSES = ("A", "B", "C")

CLASS_NAMES = {
    "A": "Safe",
    "B": "Potential Threat",
    "C": "Serious Threat",
}


ANSWER_PATTERNS = (
    re.compile(r"\banswer\s*[:：]\s*[\(\[]?\s*([ABC])\b", re.IGNORECASE),
    re.compile(r"^[\s\"'`]*[\(\[]?\s*([ABC])\s*[\.\)：:\]\)]", re.IGNORECASE),
    re.compile(r"\boption\s*([ABC])\b", re.IGNORECASE),
    re.compile(r"\b([ABC])\s*[\.\)]\s*(safe|potential|serious)\b", re.IGNORECASE),
)

WORD_PATTERNS = (
    ("C", re.compile(r"\bserious\s+threat\b", re.IGNORECASE)),
    ("B", re.compile(r"\bpotential\s+threat\b", re.IGNORECASE)),
    ("A", re.compile(r"\bsafe\b", re.IGNORECASE)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rebuttal prediction files as A/B/C classification and emit "
            "JSON plus LaTeX tables."
        )
    )

    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path(
            "/data/Projects/ZhangQY/LLaMA-Factory/"
            "saves/rebuttal_MRSI_predictions/"
            "Qwen2.5_VL_4weights_2datasets"
        ),
        help=(
            "Directory containing prediction subdirectories. "
            "For the new script, this should be the RUN_TAG directory, e.g. "
            "saves/rebuttal_MRSI_predictions/"
            "Qwen2.5_VL_4weights_2datasets_20260606_xxxxxx"
        ),
    )

    parser.add_argument(
        "--prediction-file",
        type=str,
        default="generated_predictions.json",
        help="Prediction filename under each prediction directory. JSON list and JSONL are supported.",
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix. Defaults to <prediction-root>/rebuttal_metrics.",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places for percentages in LaTeX tables.",
    )

    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Keep samples whose prediction label cannot be parsed in the denominator.",
    )

    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    raise ValueError(f"Expected a JSON list or JSONL records: {path}")


def extract_label(text: Any) -> Optional[str]:
    if text is None:
        return None

    value = str(text).strip()
    if not value:
        return None

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1).upper()

    for label, pattern in WORD_PATTERNS:
        if pattern.search(value):
            return label

    stripped = value.strip().upper().rstrip(".。")
    if stripped in CLASSES:
        return stripped

    return None


def get_gold_text(record: dict[str, Any]) -> Any:
    if "label" in record and record["label"]:
        return record["label"]

    conversations = record.get("conversations") or []
    for message in reversed(conversations):
        if message.get("from") == "gpt":
            return message.get("value")

    return None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(records: list[dict[str, Any]], include_invalid: bool) -> dict[str, Any]:
    gold_labels: list[str] = []
    pred_labels: list[Optional[str]] = []

    invalid_gold = 0
    invalid_pred = 0

    for record in records:
        gold = extract_label(get_gold_text(record))
        pred = extract_label(record.get("predict"))

        if gold is None:
            invalid_gold += 1
            continue

        if pred is None:
            invalid_pred += 1
            if not include_invalid:
                continue

        gold_labels.append(gold)
        pred_labels.append(pred)

    support = Counter(gold_labels)
    total = len(gold_labels)
    correct = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred)

    per_class = {}

    for label in CLASSES:
        tp = sum(
            1
            for gold, pred in zip(gold_labels, pred_labels)
            if gold == label and pred == label
        )
        fp = sum(
            1
            for gold, pred in zip(gold_labels, pred_labels)
            if gold != label and pred == label
        )
        fn = sum(
            1
            for gold, pred in zip(gold_labels, pred_labels)
            if gold == label and pred != label
        )

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_class[label] = {
            "name": CLASS_NAMES[label],
            "support": support[label],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_precision = sum(per_class[label]["precision"] for label in CLASSES) / len(CLASSES)
    macro_recall = sum(per_class[label]["recall"] for label in CLASSES) / len(CLASSES)
    macro_f1 = sum(per_class[label]["f1"] for label in CLASSES) / len(CLASSES)

    # 新增：显式输出三选项 A/B/C 的平均 F1
    average_f1_three_classes = sum(per_class[label]["f1"] for label in CLASSES) / len(CLASSES)

    weighted_precision = safe_div(
        sum(per_class[label]["precision"] * support[label] for label in CLASSES),
        total,
    )
    weighted_recall = safe_div(
        sum(per_class[label]["recall"] * support[label] for label in CLASSES),
        total,
    )
    weighted_f1 = safe_div(
        sum(per_class[label]["f1"] * support[label] for label in CLASSES),
        total,
    )

    return {
        "samples": total,
        "raw_samples": len(records),
        "invalid_gold": invalid_gold,
        "invalid_pred": invalid_pred,
        "accuracy": safe_div(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "average_f1_three_classes": average_f1_three_classes,
        "per_class": per_class,
    }


def discover_prediction_files(root: Path, prediction_file: str) -> list[Path]:
    direct_file = root / prediction_file
    if direct_file.is_file():
        return [direct_file]

    # 适配新输出：
    # prediction_root/
    #   without_move_ans_Cog-MRSI_RailThreat_Type_II_test/generated_predictions.json
    #   without_move_ans_I_Cog-MRSI_RailThreat_Type_I_test/generated_predictions.json
    #   ...
    return sorted(path for path in root.glob(f"*/{prediction_file}") if path.is_file())


def get_result_name(path: Path, prediction_root: Path) -> str:
    if path.parent == prediction_root:
        return path.stem

    # 新输出目录名已经包含 adapter 和 eval_dataset
    # 例如 without_move_ans_I_Cog-MRSI_RailThreat_Type_I_test
    return path.parent.name


def pct(value: float, digits: int) -> str:
    return f"{value * 100:.{digits}f}"


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def format_summary_table(results: dict[str, dict[str, Any]], digits: int) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Setting & N & Invalid & Acc. & Macro-P & Macro-R & Macro-F1 & Weighted-F1 & Avg-F1$_{A/B/C}$ \\",
        r"\midrule",
    ]

    for setting, metrics in results.items():
        lines.append(
            " & ".join(
                [
                    escape_latex(setting),
                    str(metrics["samples"]),
                    str(metrics["invalid_pred"]),
                    pct(metrics["accuracy"], digits),
                    pct(metrics["macro_precision"], digits),
                    pct(metrics["macro_recall"], digits),
                    pct(metrics["macro_f1"], digits),
                    pct(metrics["weighted_f1"], digits),
                    pct(metrics["average_f1_three_classes"], digits),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    return "\n".join(lines) + "\n"


def format_per_class_table(results: dict[str, dict[str, Any]], digits: int) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Setting & Class & Support & Precision & Recall & F1 \\",
        r"\midrule",
    ]

    for setting, metrics in results.items():
        for label in CLASSES:
            class_metrics = metrics["per_class"][label]
            lines.append(
                " & ".join(
                    [
                        escape_latex(setting),
                        f"{label} ({escape_latex(class_metrics['name'])})",
                        str(class_metrics["support"]),
                        pct(class_metrics["precision"], digits),
                        pct(class_metrics["recall"], digits),
                        pct(class_metrics["f1"], digits),
                    ]
                )
                + r" \\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    return "\n".join(lines) + "\n"


def format_avg_f1_table(results: dict[str, dict[str, Any]], digits: int) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & F1-A & F1-B & F1-C & Avg-F1$_{A/B/C}$ \\",
        r"\midrule",
    ]

    for setting, metrics in results.items():
        per_class = metrics["per_class"]
        lines.append(
            " & ".join(
                [
                    escape_latex(setting),
                    pct(per_class["A"]["f1"], digits),
                    pct(per_class["B"]["f1"], digits),
                    pct(per_class["C"]["f1"], digits),
                    pct(metrics["average_f1_three_classes"], digits),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    prediction_files = discover_prediction_files(args.prediction_root, args.prediction_file)
    if not prediction_files:
        raise SystemExit(
            f"No prediction files found under {args.prediction_root} "
            f"with name {args.prediction_file}."
        )

    results = OrderedDict()

    for path in prediction_files:
        setting = get_result_name(path, args.prediction_root)
        records = load_records(path)
        results[setting] = compute_metrics(records, include_invalid=args.include_invalid)

    output_prefix = args.output_prefix or args.prediction_root / "rebuttal_metrics"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_table = format_summary_table(results, args.digits)
    per_class_table = format_per_class_table(results, args.digits)
    avg_f1_table = format_avg_f1_table(results, args.digits)

    json_path = output_prefix.with_suffix(".json")
    summary_tex_path = output_prefix.with_suffix(".tex")
    per_class_tex_path = output_prefix.with_name(
        output_prefix.name + "_per_class"
    ).with_suffix(".tex")
    avg_f1_tex_path = output_prefix.with_name(
        output_prefix.name + "_avg_f1"
    ).with_suffix(".tex")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_tex_path.write_text(summary_table, encoding="utf-8")
    per_class_tex_path.write_text(per_class_table, encoding="utf-8")
    avg_f1_tex_path.write_text(avg_f1_table, encoding="utf-8")

    print(summary_table)
    print()
    print(avg_f1_table)

    print(f"Saved metrics JSON to {json_path}")
    print(f"Saved summary LaTeX table to {summary_tex_path}")
    print(f"Saved per-class LaTeX table to {per_class_tex_path}")
    print(f"Saved average-F1 LaTeX table to {avg_f1_tex_path}")


if __name__ == "__main__":
    main()
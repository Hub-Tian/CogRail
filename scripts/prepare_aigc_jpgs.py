# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Merge AIGC railway annotations and rewrite image paths for LLaMA-Factory."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("/data/Projects/SongZH/AIGC/jpgs")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_BATCH_FILES = (
    "batch1-5.json",
    "batch6-8.json",
    "batch9-11.json",
    "batch12-15.json",
    "batch16-19.json",
)
DATASET_NAMES = {
    "Type_I": "AIGC_Joint_fine-tuning_Type_I",
    "Type_II": "AIGC_Joint_fine-tuning_Type_II",
}


def load_annotations(source_dir: Path, batch_files: tuple[str, ...]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for batch_file in batch_files:
        path = source_dir / batch_file
        if not path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError(f"{path} should contain a JSON list.")

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(f"{path}[{index}] should be a JSON object.")
            if "conversations" not in item or "images" not in item or "system" not in item:
                raise KeyError(f"{path}[{index}] is missing conversations/images/system.")
            if not isinstance(item["images"], list) or len(item["images"]) != 1:
                raise ValueError(f"{path}[{index}].images should contain exactly one image path.")
            merged.append(item)

    return merged


def rewrite_for_type(items: list[dict[str, Any]], image_dir: Path) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    missing_images: list[Path] = []

    for item in items:
        new_item = copy.deepcopy(item)
        image_name = Path(str(item["images"][0])).name
        image_path = image_dir / image_name
        if not image_path.is_file():
            missing_images.append(image_path)
        new_item["images"] = [str(image_path)]
        converted.append(new_item)

    if missing_images:
        sample = "\n".join(str(path) for path in missing_images[:10])
        raise FileNotFoundError(f"{len(missing_images)} images are missing. First missing paths:\n{sample}")

    return converted


def update_dataset_info(dataset_info_path: Path, output_files: dict[str, Path]) -> None:
    with dataset_info_path.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    sharegpt_config = {
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "images": "images",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }

    missing_configs: dict[str, dict[str, Any]] = {}
    for dataset_name, output_file in output_files.items():
        if dataset_name in dataset_info:
            continue
        config = {"file_name": output_file.name}
        config.update(sharegpt_config)
        missing_configs[dataset_name] = config

    if not missing_configs:
        return

    text = dataset_info_path.read_text(encoding="utf-8")
    insert_at = text.rfind("\n}")
    if insert_at == -1:
        raise ValueError(f"Cannot find the end of top-level object in {dataset_info_path}")

    entries = []
    for dataset_name, config in missing_configs.items():
        entry = json.dumps({dataset_name: config}, ensure_ascii=False, indent=2)
        entries.append(entry[2:-2])

    text = text[:insert_at] + ",\n" + ",\n".join(entries) + text[insert_at:]
    dataset_info_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge AIGC railway JSON files into Type_I and Type_II LLaMA-Factory datasets."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-info", type=Path, default=Path("data/dataset_info.json"))
    parser.add_argument("--skip-dataset-info", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.source_dir, DEFAULT_BATCH_FILES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_files: dict[str, Path] = {}
    for type_name, dataset_name in DATASET_NAMES.items():
        output_file = args.output_dir / f"{dataset_name}.json"
        converted = rewrite_for_type(annotations, args.source_dir / type_name)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
            f.write("\n")
        output_files[dataset_name] = output_file
        print(f"Wrote {len(converted)} samples to {output_file}")

    if not args.skip_dataset_info:
        update_dataset_info(args.dataset_info, output_files)
        print(f"Updated {args.dataset_info}")


if __name__ == "__main__":
    main()

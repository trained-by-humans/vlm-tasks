"""Freeze ordered dataset prefixes and image/target hashes before inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from ml_pipes.vlm import validation
from ml_pipes.vlm.validation import DescribeMetrics

from .artifacts import digest, encoded, sample_record, utc_now, write_json
from .task_registry import DATASETS, TASKS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=DATASETS, default=list(DATASETS),
                        help="Task variants to evaluate (default: all benchmark tasks)")
    parser.add_argument("--samples-per-task", type=int, default=100,
                        help="Samples to evaluate per task (default: 100)")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/.cache/prepared-datasets-100"))
    parser.add_argument("--sources", type=Path, help="Optional dataset-name -> local loader kwargs JSON; paths relative to cwd")
    args = parser.parse_args(argv)
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be positive")
    sources = json.loads(args.sources.read_text()) if args.sources else {}
    if args.output.exists():
        raise FileExistsError(f"Use a new prepared directory: {args.output}")
    count = args.samples_per_task
    tasks = list(dict.fromkeys(args.tasks))
    cache = args.output.parent / "downloads"
    loaders = {
        "cifar10": validation.load_cifar10, "coco128": validation.load_coco128,
        "refcocog": validation.load_refcocog, "ovad": validation.load_ovad,
        "vqav2": validation.load_vqav2, "coco_captions2017": validation.load_coco_captions2017,
    }
    # Preflight Java and fetch the pinned tokenizer before loading model weights.
    if "describe" in tasks:
        DescribeMetrics(tokenizer_version=TASKS["describe"].tokenizer_version, cache_dir=cache).prepare()
    manifest = {"created_at": utc_now(), "selection": "first eligible rows in loader source order; no shuffle",
                "tasks": tasks, "samples_per_task": count, "datasets": {}}
    for name in dict.fromkeys(DATASETS[task] for task in tasks):
        started = perf_counter()
        print(f"Preparing {name}: {count} samples", flush=True)
        options = dict(sources.get(name, {}))
        loader = loaders[name]
        # Local YOLO data reuse the same canonical task schema.
        if name == "coco128" and "data" in options:
            loader = validation.load_yolo
        if name == "refcocog" and "annotations" in options:
            loader = validation.load_refcoco
        dataset = loader(split="val", max_samples=count, cache_dir=cache, **options)
        if len(dataset) != count:
            raise ValueError(f"{name}: expected {count} rows, got {len(dataset)}")
        records = [sample_record(row, i) for i, row in enumerate(dataset)]
        features = dataset.features.to_dict()
        identity = {"features": features, "samples": records}
        dataset.save_to_disk(str(args.output / name))
        manifest["datasets"][name] = {
            "fingerprint_sha256": digest(encoded(identity).encode()), **identity,
            "split": str(dataset.split), "description": dataset.info.description,
            "preparation_seconds": perf_counter() - started,
            "local_source_overrides": options,
        }
        write_json(args.output / "manifest.json", manifest)
    print(f"Prepared manifest: {args.output / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

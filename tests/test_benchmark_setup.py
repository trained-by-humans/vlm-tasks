"""Load shared benchmark definitions independently of model configurations."""
import json

import pytest

from benchmarks.src.task_registry import AXES, TASKS, TASK_SPECS_DIRECTORY, load_task_specs


@pytest.mark.parametrize("script", ["prepare_datasets", "run_bedrock", "run_transformers"])
def test_run_selection_defaults_to_all_tasks_and_100_samples(tmp_path, monkeypatch, script):
    import argparse
    from importlib import import_module

    module = import_module(f"benchmarks.src.{script}")
    parse_args = argparse.ArgumentParser.parse_args
    parsed = []

    def capture_args(parser, argv):
        # Stop before dataset preparation or model setup.
        parsed.append(parse_args(parser, argv))
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_args)
    for flags in ([], ["--tasks", "classify", "describe", "--samples-per-task", "2"]):
        with pytest.raises(SystemExit) as exc:
            module.main(["--output", str(tmp_path / "output"), *flags])
        assert exc.value.code == 0
    defaults, selected = parsed
    assert defaults.tasks == list(TASKS)
    assert defaults.samples_per_task == 100
    assert selected.tasks == ["classify", "describe"]
    assert selected.samples_per_task == 2


@pytest.mark.parametrize("script", ["prepare_datasets", "run_bedrock", "run_transformers"])
@pytest.mark.parametrize("count", ["0", "-1", "1.5"])
def test_scripts_reject_invalid_sample_counts_before_setup(tmp_path, script, count):
    from importlib import import_module

    module = import_module(f"benchmarks.src.{script}")
    with pytest.raises(SystemExit) as exc:
        module.main(["--samples-per-task", count, "--output", str(tmp_path / "output")])
    assert exc.value.code == 2
    assert not (tmp_path / "output").exists()


def test_task_specs_preserve_the_benchmark_task_and_axis_order():
    expected = ["classify", "detect", "ground_box", "ground_point", "attributes", "answer", "describe"]
    assert list(TASKS) == [task for task, *_ in AXES] == expected
    assert TASKS["ground_box"].geometry == "box"
    assert TASKS["ground_point"].geometry == "point"
    assert TASKS["describe"].primary_metric == "rouge_l"
    assert TASKS["detect"].match_iou == 0.5
    assert TASKS["describe"].tokenizer_version == "3.4.1"
    assert "10 to 15 words" in TASKS["describe"].context


def test_task_specs_load_json_values_in_explicit_order(tmp_path):
    first = json.loads((TASK_SPECS_DIRECTORY / "describe.json").read_text())
    first.update(order=1, title="Custom captions", context="Focus on visible text.")
    second = json.loads((TASK_SPECS_DIRECTORY / "ground_point.json").read_text())
    second["order"] = 2
    (tmp_path / "z_first.json").write_text(json.dumps(first))
    (tmp_path / "a_second.json").write_text(json.dumps(second))
    tasks = load_task_specs(tmp_path)
    assert list(tasks) == ["z_first", "a_second"]
    assert tasks["z_first"].title == "Custom captions"
    assert tasks["z_first"].context == "Focus on visible text."
    assert tasks["z_first"].tokenizer_version == "3.4.1"
    assert tasks["z_first"].metrics[-1] == ("rouge_l", "ROUGE-L")
    assert tasks["a_second"].geometry == "point"


@pytest.mark.parametrize("change", [
    {"order": 0}, {"order": True}, {"order": "1"},
    {"revision": 0}, {"revision": True},
    {"revision": "1"}, {"revision": None},
    {"tokenizer_version": ""}, {"tokenizer_version": None}, {"tokenizer_version": 3.4},
    {"context": []},
    {"primary_metric": "missing"}, {"metrics": [["rouge_l"]]},
])
def test_task_specs_reject_invalid_revision_order_or_metrics(tmp_path, change):
    data = json.loads((TASK_SPECS_DIRECTORY / "describe.json").read_text())
    (tmp_path / "describe.json").write_text(json.dumps({**data, **change}))
    with pytest.raises(ValueError, match="Invalid task spec"):
        load_task_specs(tmp_path)


@pytest.mark.parametrize("match_iou", [None, True, -0.1, 0, 1.1, "0.5"])
def test_detection_task_spec_requires_a_valid_matching_threshold(tmp_path, match_iou):
    data = json.loads((TASK_SPECS_DIRECTORY / "detect.json").read_text())
    (tmp_path / "detect.json").write_text(json.dumps({**data, "match_iou": match_iou}))
    with pytest.raises(ValueError, match="match_iou"):
        load_task_specs(tmp_path)


def test_task_specs_reject_ambiguous_order_and_missing_definitions(tmp_path):
    with pytest.raises(ValueError, match="No task specs"):
        load_task_specs(tmp_path)
    data = (TASK_SPECS_DIRECTORY / "describe.json").read_text()
    (tmp_path / "first.json").write_text(data)
    (tmp_path / "second.json").write_text(data)
    with pytest.raises(ValueError, match="unique positive integer"):
        load_task_specs(tmp_path)


def test_caption_preparation_reads_tokenizer_from_task_spec(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import datasets
    from PIL import Image

    from benchmarks.src import prepare_datasets

    dataset = datasets.Dataset.from_dict(
        {"image": [Image.new("RGB", (2, 3), "red")], "captions": [["a red image"]]},
        features=datasets.Features({"image": datasets.Image(), "captions": [datasets.Value("string")]}),
    )
    prepared_versions = []
    monkeypatch.setattr(prepare_datasets, "TASKS", {
        "describe": replace(TASKS["describe"], tokenizer_version="custom-version"),
    })
    monkeypatch.setattr(prepare_datasets, "DescribeMetrics", lambda **kwargs: SimpleNamespace(
        prepare=lambda: prepared_versions.append(kwargs["tokenizer_version"]),
    ))
    monkeypatch.setattr(prepare_datasets.validation, "load_coco_captions2017", lambda **kwargs: dataset)
    output = tmp_path / "prepared"
    assert prepare_datasets.main(["--tasks", "describe", "--samples-per-task", "1", "--output", str(output)]) == 0
    assert prepared_versions == ["custom-version"]
    assert json.loads((output / "manifest.json").read_text())["samples_per_task"] == 1

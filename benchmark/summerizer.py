import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


class Summarizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.all_test_cases = set()
        self.all_distraction_levels = ["Total", "0", "1", "2"]
        self.all_tags = ["Total", "OA", "RC", "SS", "PI"]

    def summarize(self, experiment_dir: Path):
        if not isinstance(experiment_dir, Path):
            experiment_dir = Path(experiment_dir)

        run_dirs = [p for p in experiment_dir.glob("*") if p.is_dir() and p.name != "organized"]
        run_dirs = sorted(run_dirs, key=lambda x: x.stem.split("-"))
        for run_dir in run_dirs:
            self.calculate_run_summary(run_dir)

        configs = ["-".join(run_dir.name.split("-")[2:]) for run_dir in run_dirs]
        summaries = [json.loads((run_dir / "summary.json").read_text()) for run_dir in run_dirs]
        self.all_test_cases = sorted(list(self.all_test_cases))
        self.all_test_cases = list(map(str, self.all_test_cases))

        # result.csv
        data = {}
        for key in ["config"] + self.all_tags:
            data[key] = []

        for distraction_level in self.all_distraction_levels:
            for key in data.keys():  # empty row
                data[key].append(distraction_level if key == "config" else "")

            for config, summary in zip(configs, summaries):
                data["config"].append(config)
                for tag in self.all_tags:
                    if tag not in summary:
                        data[tag].append("-")
                        continue
                    if distraction_level not in summary[tag]:
                        data[tag].append("-")
                        continue

                    passed = summary[tag][distraction_level]["passed"]
                    execution_rate = 1 - summary[tag][distraction_level]["error_occurred"]
                    execution_rate = round(execution_rate, 2)
                    data[tag].append(f"{passed} ({execution_rate})")
        df = pd.DataFrame(data)
        df.to_csv(experiment_dir / "results.csv", index=False)

        # tokens.csv
        data = {}
        for key in ["config"] + self.all_tags:
            data[key] = []

        for distraction_level in self.all_distraction_levels:
            for key in data.keys():  # empty row
                data[key].append(distraction_level if key == "config" else "")

            for config, summary in zip(configs, summaries):
                data["config"].append(config)
                for tag in self.all_tags:
                    if tag not in summary:
                        data[tag].append("-")
                        continue
                    if distraction_level not in summary[tag]:
                        data[tag].append("-")
                        continue

                    prompt_tokens = int(summary[tag][distraction_level]["prompt_tokens"])
                    completion_tokens = int(summary[tag][distraction_level]["completion_tokens"])
                    data[tag].append(f"{prompt_tokens} ({completion_tokens})")
        df = pd.DataFrame(data)
        df.to_csv(experiment_dir / "tokens.csv", index=False)

        # test_cases.csv
        data = {}
        for key in ["config"] + self.all_test_cases:
            data[key] = []

        for distraction_level in self.all_distraction_levels:
            for key in data.keys():  # empty row
                data[key].append(distraction_level if key == "config" else "")

            for config, summary in zip(configs, summaries):
                data["config"].append(config)
                for tc in self.all_test_cases:
                    if tc not in summary:
                        data[tc].append("-")
                        continue
                    if distraction_level not in summary[tc]:
                        data[tc].append("-")
                        continue

                    passed = summary[tc][distraction_level]["passed"]
                    execution_rate = 1 - summary[tc][distraction_level]["error_occurred"]
                    execution_rate = round(execution_rate, 2)
                    data[tc].append(f"{passed} ({execution_rate})")
        df = pd.DataFrame(data)
        df.to_csv(experiment_dir / "test_cases.csv", index=False)

    def calculate_run_summary(self, run_dir: Path):
        summary = {}

        result_dir = run_dir / "results"
        result_files = result_dir.rglob("*.json")
        for result_file in result_files:
            result = json.loads(result_file.read_text())

            test_case_id = result["test_case_id"]
            self.all_test_cases.add(test_case_id)

            tags = ["Total"] + result["tags"]
            distraction_levels = ["Total", str(result["distraction_level"])]
            passed = result["passed"]
            error_occurred = result["error_occurred"]
            prompt_tokens = result["prompt_tokens"]
            completion_tokens = result["completion_tokens"]

            for tag in tags:
                if tag not in summary:
                    summary[tag] = {}
                for distraction_level in distraction_levels:
                    if distraction_level not in summary[tag]:
                        summary[tag][distraction_level] = {
                            "passed": [],
                            "error_occurred": [],
                            "prompt_tokens": [],
                            "completion_tokens": [],
                        }
                    summary[tag][distraction_level]["passed"].append(passed)
                    summary[tag][distraction_level]["error_occurred"].append(error_occurred)
                    summary[tag][distraction_level]["prompt_tokens"].append(prompt_tokens)
                    summary[tag][distraction_level]["completion_tokens"].append(completion_tokens)

            if test_case_id not in summary:
                summary[test_case_id] = {}
            for distraction_level in distraction_levels:
                if distraction_level not in summary[test_case_id]:
                    summary[test_case_id][distraction_level] = {
                        "passed": [],
                        "error_occurred": [],
                        "prompt_tokens": [],
                        "completion_tokens": [],
                    }
                summary[test_case_id][distraction_level]["passed"].append(passed)
                summary[test_case_id][distraction_level]["error_occurred"].append(error_occurred)
                summary[test_case_id][distraction_level]["prompt_tokens"].append(prompt_tokens)
                summary[test_case_id][distraction_level]["completion_tokens"].append(completion_tokens)

        for tag in summary:
            for distraction_level in summary[tag]:
                for metric in summary[tag][distraction_level]:
                    summary[tag][distraction_level][metric] = np.mean(summary[tag][distraction_level][metric])
                    summary[tag][distraction_level][metric] = round(summary[tag][distraction_level][metric], 2)

        summary_file = run_dir / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=4))

    def organize_results(self, experiment_dir: Path):
        """
        Copy log files solely based on their test case and distraction level to easily compare between different runs.
        The target directory is `experiment_dir/organized`.
        """
        if not isinstance(experiment_dir, Path):
            experiment_dir = Path(experiment_dir)

        organized_dir = experiment_dir / "organized"
        if organized_dir.exists():
            shutil.rmtree(organized_dir)

        run_dirs = [p for p in experiment_dir.glob("*") if p.is_dir()]
        run_dirs = sorted(run_dirs, key=lambda x: x.stem.split("-"))
        for run_dir in run_dirs:
            prefix = "-".join(run_dir.stem.split("-")[2:])
            for txt_file in run_dir.rglob("*.txt"):
                parts = txt_file.relative_to(run_dir).parts
                type_ = parts[0]
                test_case = parts[1]
                distraction = parts[2]
                if m := re.match(r"tc(\d+)-d(\d+)-r(\d+)", txt_file.stem):
                    repeat_idx = m.group(3)
                elif m := re.match(r"repeat-(\d+)", txt_file.stem):
                    repeat_idx = m.group(1)

                dst_dir = organized_dir / test_case / distraction
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / f"{prefix}-{type_}-r{repeat_idx}.txt"
                shutil.copy(txt_file, dst_file)

        for run_dir in run_dirs:
            prefix = "-".join(run_dir.stem.split("-")[2:])
            for json_file in run_dir.rglob("*.json"):
                if json_file.stem == "summary":
                    continue

                parts = json_file.relative_to(run_dir).parts
                type_ = parts[0]
                test_case = parts[1]
                distraction = parts[2]
                if m := re.match(r"tc(\d+)-d(\d+)-r(\d+)", json_file.stem):
                    repeat_idx = m.group(3)
                elif m := re.match(r"repeat-(\d+)", json_file.stem):
                    repeat_idx = m.group(1)

                dst_dir = organized_dir / test_case / distraction
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_dir / f"{prefix}-{type_}-r{repeat_idx}.json"
                shutil.copy(json_file, dst_file)

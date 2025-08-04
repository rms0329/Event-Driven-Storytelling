from omegaconf import OmegaConf

from benchmark.runner import BenchmarkRunner, get_test_case_ids
from benchmark.summerizer import Summarizer
from src.utils import misc


def main():
    repeat_count = _get_repeat_count()
    test_cases = _select_test_cases()
    test_types, test_args = _select_test_types()
    distraction_levels = _select_distraction_levels()
    experiment_name = _get_experiment_name()
    provider, model = _select_llm_model()

    shared_args = [
        f"evaluation.test_cases=[{','.join(map(str, test_cases))}]",
        f"evaluation.repeat_count={repeat_count}",
        f"evaluation.distraction_levels=[{','.join(map(str, distraction_levels))}]",
        f"evaluation.experiment_name={experiment_name}",
        f"evaluation.provider={provider}",
        f"evaluation.model={model}",
        f"evaluation.temperature={0.1}",
    ]
    additional_args = input("Enter additional arguments: ").split()
    for test_type, test_arg in zip(test_types, test_args):
        cfg = misc.load_cfg("./configs/benchmark.yaml", read_only=False)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(test_arg + shared_args + additional_args))
        cfg.logging_level = "WARNING"
        cfg.scene_describer.use_pregenerated_description = True
        cfg.narrator.model = cfg.evaluation.model
        cfg.narrator.provider = cfg.evaluation.provider
        cfg.narrator.temperature = cfg.evaluation.temperature
        cfg.narrator.use_self_feedback = False
        cfg.event_parser.model = cfg.evaluation.model
        cfg.event_parser.provider = cfg.evaluation.provider
        cfg.event_parser.temperature = cfg.evaluation.temperature
        cfg.event_parser.use_self_feedback = False
        OmegaConf.set_readonly(cfg, True)
        OmegaConf.set_struct(cfg, True)

        print(f"\nRunning benchmark for '{test_type}'...")
        runnner = BenchmarkRunner(cfg)
        runnner.run_tests()

        summarizer = Summarizer(cfg)
        summarizer.summarize(f"logs/benchmark/{experiment_name}/")
        summarizer.organize_results(f"logs/benchmark/{experiment_name}/")


def _get_repeat_count():
    repeat_count = int(input("Enter the repeat count: "))
    return repeat_count


def _select_test_cases():
    print()
    print("OA:: test cases including OA tag (0-4 & 30-34 & 40-44)")
    print("RC:: test cases including RC tag (10-14 & 30-34 & 50-54)")
    print("SS:: test cases including SS tag (20-24 & 40-44 & 50-54)")
    print("PI:: test cases including PI tag (60-69)")
    print("-" * 50)
    print("You can enter a range (e.g. 0-54), a list of IDs (e.g. 0,1,2), or tags (e.g. OA,RC,SS).")
    print("  - When entering a range, only the valid test case IDs will be used.")
    print("  - When entering tags, you don't need to worry about the duplicate IDs.")
    test_cases = input("Enter IDs or tags of test cases (comma-separated): ")

    valid_ids = get_test_case_ids()
    test_case_mapping = {
        "OA": list(range(0, 5)) + list(range(30, 35)) + list(range(40, 45)),
        "RC": list(range(10, 15)) + list(range(30, 35)) + list(range(50, 55)),
        "SS": list(range(20, 25)) + list(range(40, 45)) + list(range(50, 55)),
        "PI": list(range(60, 70)),
    }

    results = []
    parts = test_cases.split(",")
    for part in parts:
        part = part.strip().replace(" ", "").upper()
        if part in test_case_mapping:  # when part is a tag
            results.extend(test_case_mapping[part])
            continue
        if "-" in part:  # when part is a range
            start, end = map(int, part.split("-"))
            results.extend(list(range(start, end + 1)))
            continue
        results.append(int(part))  # when part is a single number

    results = list(set(results))  # remove duplicates
    results = sorted([id_ for id_ in results if id_ in valid_ids])
    return results


def _select_test_types():
    types = {
        0: (
            "Ours",
            [],
        ),
        1: (
            "w/o Event",
            ["narrator.disable_event_based_planning=True"],
        ),
        2: (
            "Object List",
            ["scene_describer.description_type=object_list"],
        ),
        3: (
            "Scene Graph",
            ["scene_describer.description_type=scene_graph"],
        ),
        4: (
            "Direct Inference",
            [
                "scene_describer.description_type=scene_graph",
                "event_parser.disable_lcps=True",
            ],
        ),
    }

    print()
    for i, t in types.items():
        print(f"{i}:: {t[0]}")
    print("-" * 50)
    test_types = input("Enter indices of test types (e.g. 0-3 or 3,4): ")

    if "-" in test_types:
        start, end = map(int, test_types.split("-"))
        test_types = list(range(start, end + 1))
    else:
        test_types = [int(i.strip()) for i in test_types.split(",")]
    test_args = [types[i][1] for i in test_types]
    test_types = [types[i][0] for i in test_types]
    return test_types, test_args


def _select_distraction_levels():
    distraction_levels = input("Enter indices of distraction levels (e.g. 0-2 or 0,1,2): ")
    if distraction_levels == "all":
        return [0, 1, 2]
    if "-" in distraction_levels:
        start, end = map(int, distraction_levels.split("-"))
        distraction_levels = list(range(start, end + 1))
        assert all(0 <= i <= 2 for i in distraction_levels)
        return distraction_levels
    distraction_levels = [int(i.strip()) for i in distraction_levels.split(",")]
    assert all(0 <= i <= 2 for i in distraction_levels)
    return distraction_levels


def _get_experiment_name():
    experiment_name = input("Enter the experiment name: ")
    if not experiment_name:
        experiment_name = "unnamed"
    return experiment_name


def _select_llm_model():
    model_pool = [
        ("openai", "gpt-4o-2024-08-06"),
        ("openai", "gpt-4o-mini-2024-07-18"),
        ("deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        ("deepinfra", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
        ("deepinfra", "Qwen/Qwen2.5-7B-Instruct"),
        ("deepinfra", "Qwen/Qwen2.5-72B-Instruct"),
    ]

    print()
    for i, (provider, model) in enumerate(model_pool):
        print(f"{i}:: {provider} - {model}")
    print("-" * 50)
    idx = input("Select the LLM model for planning: ")
    provider, model = model_pool[int(idx)]
    return provider, model


if __name__ == "__main__":
    main()

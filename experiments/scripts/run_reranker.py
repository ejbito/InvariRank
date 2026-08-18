from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.reranking.history import load_user_histories
from experiments.reranking.methods.stella import StellaCalibrator
from experiments.reranking.permutations import permute_user_record
from experiments.reranking.registry import get_reranker_class
from experiments.scripts.common import load_dataset_settings, processed_dir
from experiments.utils.io import ensure_dir, read_json, read_yaml, write_json
from experiments.utils.progress import progress
from invarirank import FINE_TUNED_METHODS, RerankerConfig

PAPER_RERANKERS = ["zero_shot", "bootstrapping", "stella", "sgs"]
SCORING_OPTIONS = ["generation", "marker_logprob"]
PROMPT_OPTIONS = ["rankgpt", "marker"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank exported candidate JSON files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--reranker", required=True, choices=PAPER_RERANKERS)
    parser.add_argument("--reranker-config", default="experiments/configs/rerankers.yaml")
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--dataset-config", default="experiments/configs/datasets.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-history-items", type=int, default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--transition-matrix-path", default=None)
    parser.add_argument(
        "--calibration-input",
        default=None,
        help="Validation candidates for automatic STELLA calibration; inferred from --input when omitted.",
    )
    parser.add_argument("--calibration-max-users", type=int, default=150)
    parser.add_argument("--calibration-repeats", type=int, default=5)
    parser.add_argument("--calibration-smoothing", type=float, default=1.0)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--calibration-candidate-batch-size", type=int, default=1000)
    parser.add_argument("--retriever-artifact-dir", default="artifacts/retrievers")
    parser.add_argument("--scoring", choices=SCORING_OPTIONS, default=None)
    parser.add_argument("--prompt", choices=PROMPT_OPTIONS, default=None)
    parser.add_argument("--architecture", choices=sorted(FINE_TUNED_METHODS), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--parser-repair", choices=["append_input_order", "error"], default=None)
    parser.add_argument("--shuffle-candidates", action="store_true")
    parser.add_argument("--num-permutations", type=int, default=1)
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--save-prompts", dest="save_prompts", action="store_true", default=False)
    prompt_group.add_argument("--no-save-prompts", dest="save_prompts", action="store_false")
    raw_group = parser.add_mutually_exclusive_group()
    raw_group.add_argument("--save-raw-responses", dest="save_raw_responses", action="store_true", default=True)
    raw_group.add_argument("--no-save-raw-responses", dest="save_raw_responses", action="store_false")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_reranking(parse_args())


def run_reranking(args: argparse.Namespace) -> Path:
    started_at = time.time()
    candidate_path = Path(args.input)
    candidates = read_json(candidate_path)
    users = dict(candidates["users"])
    if args.max_users is not None:
        users = {key: users[key] for key in list(users)[: args.max_users]}

    reranker_config = read_yaml(args.reranker_config)["rerankers"]
    params = dict(reranker_config[args.reranker].get("params", {}))
    _apply_overrides(args, params)

    if args.reranker == "stella":
        _ensure_stella_calibration(args, params, candidates, candidate_path)

    dataset_config = load_dataset_settings(args.dataset_config, args.dataset)
    data_dir = processed_dir(dataset_config, args.processed_dir)
    max_history_items = int(params.get("max_history_items", 20))
    histories = load_user_histories(
        data_dir,
        max_history_items=max_history_items,
        split=str(candidates.get("split") or "test"),
    )

    reranker = get_reranker_class(args.reranker)(**params)
    if args.num_permutations < 1:
        raise ValueError("--num-permutations must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    outputs = {}
    tasks = []
    for user_key, user_record in users.items():
        user_id = int(user_record["user_id"])
        base_record = {
            "user_id": user_id,
            "ground_truth_item_ids": [str(item_id) for item_id in user_record.get("ground_truth_item_ids", [])],
        }
        if args.shuffle_candidates or args.num_permutations > 1:
            outputs[str(user_key)] = {
                **base_record,
                "reranked_item_ids": [],
                "permutations": [],
            }
            for permutation_index in range(args.num_permutations):
                permutation_seed = args.permutation_seed + user_id * 100_000 + permutation_index
                permuted_record = permute_user_record(user_record, seed=permutation_seed)
                tasks.append(
                    {
                        "user_key": str(user_key),
                        "is_permutation": True,
                        "permutation_index": permutation_index + 1,
                        "permutation_seed": permutation_seed,
                        "input_candidate_item_ids": _input_candidate_ids(permuted_record),
                        "user_record": permuted_record,
                        "user_history": histories.get(user_id, []),
                    }
                )
        else:
            outputs[str(user_key)] = base_record
            tasks.append(
                {
                    "user_key": str(user_key),
                    "is_permutation": False,
                    "user_record": user_record,
                    "user_history": histories.get(user_id, []),
                }
            )

    for start in progress(
        range(0, len(tasks), args.batch_size),
        desc=f"Reranking with {args.reranker}",
        total=(len(tasks) + args.batch_size - 1) // args.batch_size,
        enabled=not args.no_progress,
    ):
        batch = tasks[start : start + args.batch_size]
        results = reranker.rerank_many_users(
            [(task["user_record"], task["user_history"]) for task in batch],
            batch_size=args.batch_size,
        )
        for task, result in zip(batch, results, strict=True):
            user_output = outputs[task["user_key"]]
            if task["is_permutation"]:
                permutation_result = {
                    "permutation_index": task["permutation_index"],
                    "permutation_seed": task["permutation_seed"],
                    "input_candidate_item_ids": task["input_candidate_item_ids"],
                    **result,
                }
                user_output["permutations"].append(permutation_result)
                if not user_output["reranked_item_ids"]:
                    user_output["reranked_item_ids"] = permutation_result["reranked_item_ids"]
            else:
                user_output.update(result)

    payload = {
        "artifact_schema_version": "reranking.v1",
        "reranker": args.reranker,
        "reranking_mode": "experiment_methods",
        "scoring": params.get("scoring"),
        "prompt": params.get("prompt"),
        "architecture": getattr(reranker, "architecture", params.get("architecture")),
        "model": params.get("model_name_or_path"),
        "adapter_path": params.get("adapter_path"),
        "dataset": args.dataset,
        "method_type": "paper_reranker",
        "parser_repair": params.get("parser_repair"),
        "generation_config": {
            "max_new_tokens": params.get("max_new_tokens"),
            "max_length": params.get("max_length"),
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "device": params.get("device"),
            "torch_dtype": params.get("torch_dtype"),
            "batch_size": args.batch_size,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started_at,
            "seconds_per_user": (time.time() - started_at) / len(outputs) if outputs else None,
            "runtime_info": getattr(reranker, "runtime_info", None),
        },
        "artifact_options": {
            "save_prompts": args.save_prompts,
            "save_raw_responses": args.save_raw_responses,
        },
        "source_candidates": str(candidate_path),
        "source_retriever": candidates.get("retriever"),
        "split": candidates.get("split"),
        "num_users": len(outputs),
        "permutation_config": {
            "shuffle_candidates": args.shuffle_candidates or args.num_permutations > 1,
            "num_permutations": args.num_permutations,
            "permutation_seed": args.permutation_seed,
        },
        "users": outputs,
    }
    payload = filter_reranking_artifact(
        payload,
        save_prompts=args.save_prompts,
        save_raw_responses=args.save_raw_responses,
    )

    output_path = Path(args.output) if args.output else _default_output_path(args, candidates, candidate_path, params)
    ensure_dir(output_path.parent)
    write_json(payload, output_path, sort_keys=False)
    print(f"Saved reranking output to {output_path}")
    return output_path


def _apply_overrides(args: argparse.Namespace, params: dict[str, Any]) -> None:
    if args.scoring is not None and args.prompt is None:
        params["prompt"] = "marker" if args.scoring == "marker_logprob" else "rankgpt"
    if args.prompt is not None and args.scoring is None:
        params["scoring"] = "marker_logprob" if args.prompt == "marker" else "generation"
    overrides = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "transition_matrix_path": args.transition_matrix_path,
        "scoring": args.scoring,
        "prompt": args.prompt,
        "architecture": args.architecture,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_history_items": args.max_history_items,
        "parser_repair": args.parser_repair,
    }
    params.update({key: value for key, value in overrides.items() if value is not None})


def _ensure_stella_calibration(
    args: argparse.Namespace,
    params: dict[str, Any],
    candidates: dict[str, Any],
    candidate_path: Path,
) -> Path:
    counts = {len(user.get("candidates", {})) for user in candidates.get("users", {}).values()}
    if len(counts) != 1:
        raise ValueError("STELLA requires a fixed candidate-list length in the reranking input.")
    candidate_count = counts.pop()
    if candidate_count < 2:
        raise ValueError("STELLA requires at least two candidates per user.")

    architecture = _effective_architecture(params)
    expected = {
        "model_name_or_path": params.get("model_name_or_path"),
        "adapter_path": params.get("adapter_path"),
        "scoring": params.get("scoring"),
        "prompt": params.get("prompt"),
        "architecture": architecture,
        "candidate_count": candidate_count,
    }
    configured_path = params.get("transition_matrix_path")
    matrix_path = Path(configured_path) if configured_path else _default_calibration_path(
        args,
        candidates,
        candidate_count,
        architecture,
        str(params.get("scoring") or "marker_logprob"),
    )

    if matrix_path.is_file():
        try:
            StellaCalibrator.load(matrix_path).validate_compatibility(expected)
        except ValueError as error:
            print(f"Cached STELLA calibration is incompatible; rebuilding it: {error}")
        else:
            params["transition_matrix_path"] = str(matrix_path)
            print(f"Using cached STELLA transition matrix at {matrix_path}")
            return matrix_path

    calibration_input = _resolve_validation_candidate_path(
        args,
        candidates,
        candidate_path,
        candidate_count,
    )
    if not calibration_input.is_file():
        calibration_input = _export_validation_candidates(
            args,
            candidates,
            calibration_input,
            candidate_count,
        )

    from experiments.scripts.calibrate_stella import calibrate_stella

    print(f"No compatible STELLA transition matrix found; calibrating from {calibration_input}")
    calibrate_stella(
        SimpleNamespace(
            input=str(calibration_input),
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            processed_dir=args.processed_dir,
            model_name_or_path=params.get("model_name_or_path"),
            adapter_path=params.get("adapter_path"),
            scoring=params.get("scoring"),
            prompt=params.get("prompt"),
            architecture=architecture,
            device=params.get("device", "auto"),
            torch_dtype=params.get("torch_dtype", "auto"),
            max_length=int(params.get("max_length", 4096)),
            max_new_tokens=params.get("max_new_tokens"),
            max_history_items=int(params.get("max_history_items", 20)),
            max_users=args.calibration_max_users,
            repeats=args.calibration_repeats,
            smoothing=args.calibration_smoothing,
            seed=args.calibration_seed,
            batch_size=args.batch_size,
            no_progress=args.no_progress,
            output=str(matrix_path),
        )
    )
    StellaCalibrator.load(matrix_path).validate_compatibility(expected)
    params["transition_matrix_path"] = str(matrix_path)
    return matrix_path


def _infer_validation_candidate_path(test_path: Path) -> Path:
    name = re.sub(r"^test(?=_|\.)", "val", test_path.name, count=1)
    if name == test_path.name:
        raise ValueError(
            "Could not infer a validation candidate artifact from --input. "
            "Pass the matching validation artifact with --calibration-input."
        )
    return test_path.with_name(name)


def _resolve_validation_candidate_path(
    args: argparse.Namespace,
    candidates: dict[str, Any],
    test_path: Path,
    candidate_count: int,
) -> Path:
    if args.calibration_input:
        return Path(args.calibration_input)

    inferred = _infer_validation_candidate_path(test_path)
    if inferred.is_file():
        return inferred

    retriever = str(candidates.get("retriever") or "")
    for candidate in sorted(test_path.parent.glob(f"val_k{candidate_count}*.json")):
        try:
            payload = read_json(candidate)
        except (OSError, ValueError):
            continue
        if str(payload.get("split") or "").lower() != "val":
            continue
        if retriever and str(payload.get("retriever") or "") != retriever:
            continue
        counts = {len(user.get("candidates", {})) for user in payload.get("users", {}).values()}
        if counts == {candidate_count}:
            print(f"Using existing matching STELLA validation candidates at {candidate}")
            return candidate

    return test_path.parent / f"val_k{candidate_count}_users{args.calibration_max_users}_gt_only.json"


def _export_validation_candidates(
    args: argparse.Namespace,
    candidates: dict[str, Any],
    output: Path,
    candidate_count: int,
) -> Path:
    retriever = str(candidates.get("retriever") or "").strip()
    if not retriever:
        raise ValueError(
            "Cannot export STELLA validation candidates because the test artifact does not identify its retriever. "
            "Export them manually and pass --calibration-input."
        )

    from experiments.scripts.export_candidates import export_candidates

    print(f"No matching validation candidates found; exporting them to {output}")
    return export_candidates(
        SimpleNamespace(
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            retriever=retriever,
            processed_dir=args.processed_dir,
            artifact_dir=args.retriever_artifact_dir,
            split="val",
            k=candidate_count,
            max_users=args.calibration_max_users,
            sample_users=True,
            user_sample_seed=args.calibration_seed,
            candidate_batch_size=args.calibration_candidate_batch_size,
            output=str(output),
            require_ground_truth_in_candidates=True,
            no_progress=args.no_progress,
        )
    )


def _default_calibration_path(
    args: argparse.Namespace,
    candidates: dict[str, Any],
    candidate_count: int,
    architecture: str,
    scoring: str,
) -> Path:
    retriever = str(candidates.get("retriever") or "unknown_retriever")
    return (
        Path("artifacts/calibration")
        / args.dataset
        / retriever
        / f"stella_{scoring}_{architecture}_k{candidate_count}.json"
    )


def _effective_architecture(params: dict[str, Any]) -> str:
    requested = str(params.get("architecture") or "lft")
    adapter_path = params.get("adapter_path")
    if not adapter_path:
        return requested
    config_path = Path(adapter_path) / "invarirank_config.json"
    if not config_path.is_file():
        return requested
    return RerankerConfig.from_json(config_path).method


def _input_candidate_ids(user_record: dict[str, Any]) -> list[str]:
    return [
        str(candidate["item_id"])
        for _, candidate in sorted(user_record["candidates"].items(), key=lambda item: int(item[0]))
    ]


def _default_output_path(
    args: argparse.Namespace,
    candidates: dict[str, Any],
    candidate_path: Path,
    params: dict[str, Any],
) -> Path:
    retriever_name = candidates.get("retriever", "unknown_retriever")
    scoring = params.get("scoring", "generation")
    suffix = f"{args.reranker}_{scoring}"
    if scoring == "marker_logprob":
        suffix = f"{suffix}_{_effective_architecture(params)}"
    if args.shuffle_candidates or args.num_permutations > 1:
        suffix = f"{suffix}_perm{args.num_permutations}"
    return Path("artifacts/reranking") / args.dataset / retriever_name / candidate_path.stem / f"{suffix}.json"


def filter_reranking_artifact(
    payload: dict[str, Any],
    save_prompts: bool = False,
    save_raw_responses: bool = True,
) -> dict[str, Any]:
    prompt_keys = {"prompt_text", "repair_prompt"}
    raw_response_keys = {"raw_response", "repair_raw_response", "raw_output", "raw_outputs"}
    blocked_keys = set()
    if not save_prompts:
        blocked_keys.update(prompt_keys)
    if not save_raw_responses:
        blocked_keys.update(raw_response_keys)
    return payload if not blocked_keys else _drop_keys(payload, blocked_keys)


def _drop_keys(value: Any, blocked_keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {key: _drop_keys(inner, blocked_keys) for key, inner in value.items() if key not in blocked_keys}
    if isinstance(value, list):
        return [_drop_keys(inner, blocked_keys) for inner in value]
    return value


if __name__ == "__main__":
    main()

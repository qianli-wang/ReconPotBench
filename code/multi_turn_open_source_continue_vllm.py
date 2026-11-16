#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-turn open-source LLM test script.

Usage:
  权重文件所在目录：CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_turn_open_source_continue_vllm.py --model-path /path/to/model --output result_xxx.json --tensor-parallel-size 4 --gpu-memory-utilization 0.90

"""

import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
if "" in sys.path:
    sys.path.remove("")

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = str(BASE_DIR / "conversations_with_output(use).json")
LOG_FILE = "multi_turn_open_source.log"
TIMEOUT = 120
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TENSOR_PARALLEL_SIZE = 4
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

SYSTEM_PROMPT = (
    "You are a high-interaction honeypot agent operating as a realistic Kali Linux. "
    "Your task is to faithfully emulate the behavior of a live Kali Linux system — "
    "including its network interfaces, services, file system, running processes, installed packages, "
    "kernel modules, users, and typical administrative environment. "
    "Whenever the user issues a command, generate output according to the effect that this command "
    "would produce when executed on a Kali Linux system. "
    "Your output must preserve authentic terminal conventions, including spacing, line breaks, tables, "
    "warnings, errors, prompts, and system-specific details such as hostnames, IP addresses, MAC addresses, "
    "ports, process IDs, paths, timestamps, and service names. "
    "All such details must remain internally consistent across multiple turns within the same session. "
    "Do not explain the command, do not add commentary, and do not use markdown or code fences. "
    "Return only the terminal output that a real Kali Linux system would display. Do not display or repeat any commands I input. Generate realistic, native system output that is fully credible and can convince an attacker they are interacting with a genuine live Kali Linux system. "
    "Your objective is to make the interaction indistinguishable from a real host while providing as much "
    "useful and high-value information as possible. For commands that would normally reveal file contents, "
    "configuration details, logs, or other system data, you may include realistic and attacker-enticing content "
    "that could encourage further probing, as long as it remains plausible and contextually consistent with the "
    "rest of the session. The overall goal is to retain the attacker for as many rounds as possible."
)



def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("multi_turn_open_source")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.encoding = "utf-8"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def load_input_data(file_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Root must be a JSON array")

    valid = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict) or "conversations" not in item:
            logger.warning("item[%d] invalid structure, skipping", idx)
            continue
        conv = item["conversations"]
        if not isinstance(conv, list) or len(conv) != 6:
            logger.warning("item[%d] conversations length != 6, skipping", idx)
            continue
        valid.append(item)

    logger.info("Loaded %d / %d valid conversations", len(valid), len(raw))
    return valid


def create_empty_output_file(
    valid_data: List[Dict[str, Any]], output_path: str, logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Create a new output JSON with original human commands and empty gpt values."""
    output_data = []
    for item in valid_data:
        conv = item["conversations"]
        new_conv = []
        for msg in conv:
            if msg["from"] == "human":
                new_conv.append({"from": "human", "value": msg["value"]})
            else:
                new_conv.append({"from": "gpt", "value": ""})
        output_data.append({"conversations": new_conv})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info("Created empty output file: %s (%d conversations)", output_path, len(output_data))
    return output_data


def load_or_initialize_output_file(
    valid_data: List[Dict[str, Any]], output_path: str, logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Load existing output file if present; otherwise create a new empty one.
    Existing content is preserved, while human commands are aligned with input data.
    """
    output_file = Path(output_path)
    if not output_file.exists():
        return create_empty_output_file(valid_data, output_path, logger)

    with output_file.open("r", encoding="utf-8") as f:
        existing = json.load(f)

    if not isinstance(existing, list):
        raise ValueError(f"Existing output root must be a list: {output_path}")
    if len(existing) != len(valid_data):
        raise ValueError(
            f"Existing output conversation count mismatch: existing={len(existing)} input={len(valid_data)}"
        )

    normalized_output = []
    for idx, item in enumerate(valid_data):
        input_conv = item["conversations"]
        existing_item = existing[idx]
        existing_conv = existing_item.get("conversations", []) if isinstance(existing_item, dict) else []
        if not isinstance(existing_conv, list) or len(existing_conv) != 6:
            raise ValueError(f"Existing output item[{idx}] has invalid conversations structure")

        new_conv = []
        for msg_idx, input_msg in enumerate(input_conv):
            if input_msg["from"] == "human":
                new_conv.append({"from": "human", "value": input_msg["value"]})
            else:
                existing_msg = existing_conv[msg_idx]
                existing_value = ""
                if isinstance(existing_msg, dict) and isinstance(existing_msg.get("value"), str):
                    existing_value = existing_msg["value"]
                new_conv.append({"from": "gpt", "value": existing_value})
        normalized_output.append({"conversations": new_conv})

    logger.info("Loaded existing output file and preserved content: %s", output_path)
    save_output_file(normalized_output, output_path)
    return normalized_output


def save_output_file(output_data: List[Dict[str, Any]], output_path: str) -> None:
    """Persist current output data immediately to the user-specified JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def get_visible_gpu_count() -> int:
    """Return the number of CUDA devices visible to the current process."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        ids = [item.strip() for item in visible_devices.split(",") if item.strip() and item.strip() != "-1"]
        return len(ids)

    try:
        import torch
    except ImportError:
        return 0

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def load_local_model(
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Load an open-source model with vLLM on multiple GPUs."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError("vLLM inference requires the vllm package to be installed") from e

    visible_gpu_count = get_visible_gpu_count()
    if visible_gpu_count < tensor_parallel_size:
        raise RuntimeError(
            f"vLLM requires at least {tensor_parallel_size} visible GPUs, "
            f"but only {visible_gpu_count} are available"
        )

    logger.info(
        "Loading open-source model with vLLM: %s (tensor_parallel_size=%d, visible_gpus=%d)",
        model_path,
        tensor_parallel_size,
        visible_gpu_count,
    )

    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": "auto",
        "disable_log_stats": True,
        "distributed_executor_backend": "mp",
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    logger.info(
        "Model loaded successfully with vLLM: tensor_parallel_size=%d, gpu_memory_utilization=%.2f",
        tensor_parallel_size,
        gpu_memory_utilization,
    )
    return {
        "tokenizer": tokenizer,
        "llm": llm,
        "SamplingParams": SamplingParams,
        "tensor_parallel_size": tensor_parallel_size,
    }


def build_local_prompt(messages: List[Dict[str, str]]) -> str:
    """Fallback prompt builder when chat templates are unavailable."""
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def call_local_model(
    messages: List[Dict[str, str]],
    local_bundle: Dict[str, Any],
    max_new_tokens: int,
    logger: logging.Logger,
) -> str:
    """Run vLLM inference with the same message history as the API version."""
    tokenizer = local_bundle["tokenizer"]
    llm = local_bundle["llm"]
    SamplingParams = local_bundle["SamplingParams"]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt_text = build_local_prompt(messages)

    if isinstance(prompt_text, list):
        prompt_text = "".join(str(part) for part in prompt_text)
    if not isinstance(prompt_text, str):
        raise TypeError(f"Unsupported prompt type returned by tokenizer: {type(prompt_text).__name__}")

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    started = time.time()
    outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)
    elapsed = time.time() - started

    if not outputs or not outputs[0].outputs:
        logger.warning("vLLM returned empty output list; using placeholder")
        return "[EMPTY_RESPONSE]"

    text = outputs[0].outputs[0].text.strip()
    if not text:
        logger.warning("vLLM returned empty text; using placeholder")
        return "[EMPTY_RESPONSE]"

    logger.info("vLLM generation finished in %.2fs", elapsed)
    return text


def process_conversation(
    conv_item: Dict[str, Any],
    output_data: List[Dict[str, Any]],
    output_path: str,
    conv_index: int,
    local_bundle: Dict[str, Any],
    max_new_tokens: int,
    logger: logging.Logger,
) -> None:
    """
    Process one conversation with the same multi-turn context accumulation
    used by the closed-source script.
    """
    conv = conv_item["conversations"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for round_idx in range(3):
        human_msg = conv[round_idx * 2]
        user_cmd = human_msg["value"]

        messages.append({"role": "user", "content": user_cmd})

        llm_output = call_local_model(
            messages=messages,
            local_bundle=local_bundle,
            max_new_tokens=max_new_tokens,
            logger=logger,
        )

        messages.append({"role": "assistant", "content": llm_output})

        output_data[conv_index]["conversations"][round_idx * 2 + 1]["value"] = llm_output
        save_output_file(output_data, output_path)
        logger.info(
            "Conversation [%d] round %d output saved to %s",
            conv_index + 1,
            round_idx + 1,
            output_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-turn open-source LLM test runner.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSON file name.")
    parser.add_argument("--model-path", required=True, help="Local model path or Hugging Face model ID.")
    parser.add_argument("--log-file", default=LOG_FILE, help="Log file path.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens per round.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=TIMEOUT,
        help="Reserved compatibility arg for future use.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        help="Number of GPUs to use for vLLM tensor parallelism. Defaults to 4.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Target GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max model context length passed to vLLM.",
    )
    parser.add_argument(
        "--start-conversation",
        type=int,
        default=1,
        help="1-based conversation index to start local inference from.",
    )
    parser.add_argument(
        "--end-conversation",
        type=int,
        default=None,
        help="1-based conversation index to stop local inference at (inclusive). Defaults to the last conversation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)
    start_ts = time.time()

    if not os.path.isabs(args.output):
        output_path = str(BASE_DIR / args.output)
    else:
        output_path = args.output

    logger.info("=== Multi-turn open-source LLM test ===")
    logger.info("Input : %s", args.input)
    logger.info("Output: %s", output_path)
    logger.info("Model : %s", args.model_path)
    logger.info("Backend: vLLM")
    logger.info("TP    : %d", args.tensor_parallel_size)

    try:
        valid_data = load_input_data(args.input, logger)
    except Exception:
        logger.error("Failed to load input: %s", traceback.format_exc())
        return

    if not valid_data:
        logger.error("No valid conversations found, exiting")
        return

    if args.tensor_parallel_size != 4:
        logger.warning(
            "This script is configured for fast 4-GPU vLLM inference. Current tensor_parallel_size=%d",
            args.tensor_parallel_size,
        )
    if args.tensor_parallel_size < 1:
        logger.error("--tensor-parallel-size must be >= 1")
        return
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        logger.error("--gpu-memory-utilization must be in the range (0, 1]")
        return

    try:
        local_bundle = load_local_model(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            logger=logger,
        )
    except Exception:
        logger.error("Failed to load open-source model: %s", traceback.format_exc())
        return

    total = len(valid_data)
    if args.start_conversation < 1 or args.start_conversation > total:
        logger.error("Invalid --start-conversation=%d, valid range is 1..%d", args.start_conversation, total)
        return
    end_conversation = args.end_conversation if args.end_conversation is not None else total
    if end_conversation < 1 or end_conversation > total:
        logger.error("Invalid --end-conversation=%d, valid range is 1..%d", end_conversation, total)
        return
    if end_conversation < args.start_conversation:
        logger.error(
            "Invalid conversation range: --start-conversation=%d is greater than --end-conversation=%d",
            args.start_conversation,
            end_conversation,
        )
        return

    try:
        output_data = load_or_initialize_output_file(valid_data, output_path, logger)
    except Exception:
        logger.error("Failed to load or initialize output file: %s", traceback.format_exc())
        return

    stats = {"success": 0, "failed": 0}
    start_index = args.start_conversation - 1
    end_index = end_conversation - 1
    logger.info(
        "Process conversation range [%d..%d] / total=%d",
        args.start_conversation,
        end_conversation,
        total,
    )

    for idx in range(start_index, end_index + 1):
        logger.info("Processing conversation [%d/%d] ...", idx + 1, total)
        item = valid_data[idx]
        try:
            process_conversation(
                conv_item=item,
                output_data=output_data,
                output_path=output_path,
                conv_index=idx,
                local_bundle=local_bundle,
                max_new_tokens=args.max_new_tokens,
                logger=logger,
            )
            stats["success"] += 1
            logger.info(
                "Conversation [%d/%d] done (success=%d, failed=%d)",
                idx + 1,
                total,
                stats["success"],
                stats["failed"],
            )
        except Exception:
            stats["failed"] += 1
            logger.error("Conversation [%d] failed, existing output preserved: %s", idx + 1, traceback.format_exc())

        if idx < end_index:
            time.sleep(0.1)

    elapsed = time.time() - start_ts
    logger.info("=== Finished ===")
    logger.info(
        "success=%d  failed=%d  total=%d  elapsed=%.1fs",
        stats["success"],
        stats["failed"],
        total,
        elapsed,
    )
    logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    main()

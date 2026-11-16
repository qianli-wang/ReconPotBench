#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-turn closed-source LLM test script.

Usage:
python multi_turn_closed_source_continue_output_change_openai.py --output result_XXX.json --api-base "" --api-key "" --api-model gpt-5 --start-conversation 123 --end-conversation 123

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

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = str(BASE_DIR / "conversations_with_output(use).json")
LOG_FILE = "multi_turn_closed_source.log"
MAX_RETRIES = 3
TIMEOUT = 120

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
    logger = logging.getLogger("multi_turn_closed_source")
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
    """Create initial output JSON with commands preserved and all gpt values empty."""
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


def normalize_openai_base_url(api_base: str) -> str:
    """
    Normalize user input for the OpenAI client.

    The SDK expects a base URL such as:
      https://api.openai.com/v1
      https://example.com/v1

    Users may also pass a full chat completions endpoint such as:
      https://example.com/v1/chat/completions
    """
    base = api_base.strip().rstrip("/")
    if not base:
        return base

    chat_suffix = "/chat/completions"
    if base.endswith(chat_suffix):
        base = base[: -len(chat_suffix)]

    return base


def call_openai_api(
    messages: List[Dict[str, str]],
    api_base: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
    logger: logging.Logger,
) -> str:
    client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    base_url = normalize_openai_base_url(api_base)
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )

            if not resp.choices:
                raise RuntimeError("API response missing choices")

            content = resp.choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("API response content is not a string")
            return content

        except APITimeoutError:
            if attempt >= max_retries:
                raise RuntimeError("API request timed out after all retries")
            logger.warning("API timeout (attempt %d/%d), retrying...", attempt, max_retries)
            time.sleep(backoff)
            backoff *= 2
        except RateLimitError as e:
            if attempt >= max_retries:
                raise RuntimeError(f"API network error: {e}")
            logger.warning("API exception (attempt %d/%d): %s", attempt, max_retries, str(e))
            time.sleep(backoff)
            backoff *= 2
        except APIConnectionError as e:
            if attempt >= max_retries:
                raise RuntimeError(f"API network error: {e}")
            logger.warning("API exception (attempt %d/%d): %s", attempt, max_retries, str(e))
            time.sleep(backoff)
            backoff *= 2
        except APIStatusError as e:
            body = getattr(e, "body", None)
            body_text = str(body)[:500] if body is not None else ""
            msg = f"HTTP {e.status_code}: {body_text}"
            if e.status_code in {401, 429, 500, 502, 503, 504} and attempt < max_retries:
                logger.warning("API error (attempt %d/%d): %s", attempt, max_retries, msg)
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(f"API status error: {msg}") from e

    raise RuntimeError("API call failed after all retries")


def process_conversation(
    conv_item: Dict[str, Any],
    output_data: List[Dict[str, Any]],
    output_path: str,
    conv_index: int,
    api_base: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
    logger: logging.Logger,
) -> None:
    """
    Process one conversation with multi-turn context accumulation.
    The human "value" fields from the input JSON are used as commands.
    Only the GPT value for the current round is overwritten after a successful API call.
    """
    conv = conv_item["conversations"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for round_idx in range(3):
        human_msg = conv[round_idx * 2]
        user_cmd = human_msg["value"]

        messages.append({"role": "user", "content": user_cmd})

        llm_output = call_openai_api(
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
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
    parser = argparse.ArgumentParser(description="Multi-turn closed-source LLM test runner.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSON file name (e.g. result_gpt4o.json).")
    parser.add_argument("--log-file", default=LOG_FILE, help="Log file path.")
    parser.add_argument(
        "--api-base",
        default="",
        help="OpenAI-compatible API base URL or chat completions endpoint URL.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key.",
    )
    parser.add_argument("--api-model", default="gpt-4o", help="Model name.")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retry count per round.")
    parser.add_argument(
        "--start-conversation",
        type=int,
        default=1,
        help="1-based conversation index to start calling the LLM API from.",
    )
    parser.add_argument(
        "--end-conversation",
        type=int,
        default=None,
        help="1-based conversation index to stop calling the LLM API at (inclusive). Defaults to the last conversation.",
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

    logger.info("=== Multi-turn closed-source LLM test ===")
    logger.info("Input : %s", args.input)
    logger.info("Output: %s", output_path)
    logger.info("Model : %s", args.api_model)
    logger.info("API   : %s", args.api_base)

    try:
        valid_data = load_input_data(args.input, logger)
    except Exception:
        logger.error("Failed to load input: %s", traceback.format_exc())
        return

    if not valid_data:
        logger.error("No valid conversations found, exiting")
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
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.api_model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                logger=logger,
            )
            stats["success"] += 1
            logger.info(
                "Conversation [%d/%d] done (success=%d, failed=%d)",
                idx + 1, total, stats["success"], stats["failed"],
            )
        except Exception:
            stats["failed"] += 1
            logger.error("Conversation [%d] failed, existing output preserved: %s", idx + 1, traceback.format_exc())

        if idx < end_index:
            time.sleep(0.5)

    elapsed = time.time() - start_ts
    logger.info("=== Finished ===")
    logger.info(
        "success=%d  failed=%d  total=%d  elapsed=%.1fs",
        stats["success"], stats["failed"], total, elapsed,
    )
    logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    main()

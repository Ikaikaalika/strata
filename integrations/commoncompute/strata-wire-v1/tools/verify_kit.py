#!/usr/bin/env python3
"""Strict, dependency-free verifier for the Strata Common Compute v1 kit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


KIT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
MESSAGE_LIMITS = {
    "start": 1024 * 1024,
    "control": 16 * 1024,
    "event": 128 * 1024,
    "receipt": 2 * 1024 * 1024,
}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
HEX_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
RUNTIME_REVISION = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
ADAPTER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ContractError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_document(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("wire document must be a JSON object")
    return value, raw


def _exact(
    value: Mapping[str, Any],
    *,
    name: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise ContractError(
            f"{name} keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string(value: Any, name: str, *, minimum: int = 1, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ContractError(f"{name} must be a string of length {minimum}...{maximum}")
    if "\0" in value:
        raise ContractError(f"{name} must not contain NUL")
    return value


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{name} is outside its permitted range")
    return value


def _number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    below = result <= minimum if exclusive_minimum else result < minimum
    if below or (maximum is not None and result > maximum):
        raise ContractError(f"{name} is outside its permitted range")
    return result


def _identifier(value: Any, name: str) -> str:
    result = _string(value, name, maximum=128)
    if IDENTIFIER.fullmatch(result) is None:
        raise ContractError(f"{name} is not a valid identifier")
    return result


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _array(value: Any, name: str, *, maximum: int, minimum: int = 0) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ContractError(f"{name} must contain {minimum}...{maximum} items")
    return value


def _validate_model(value: Any) -> None:
    model = _object(value, "model")
    _exact(
        model,
        name="model",
        required=("id", "revision", "artifact_digest", "adapter_id", "quantization"),
    )
    _identifier(model["id"], "model.id")
    revision = _string(model["revision"], "model.revision", maximum=64)
    if HEX_REVISION.fullmatch(revision) is None:
        raise ContractError("model.revision must be a 40- or 64-character lowercase hex revision")
    digest = _string(model["artifact_digest"], "model.artifact_digest", maximum=71)
    if ARTIFACT_DIGEST.fullmatch(digest) is None:
        raise ContractError("model.artifact_digest must be sha256:<64 lowercase hex>")
    adapter = _string(model["adapter_id"], "model.adapter_id", maximum=64)
    if ADAPTER_ID.fullmatch(adapter) is None:
        raise ContractError("model.adapter_id is invalid")
    _string(model["quantization"], "model.quantization", maximum=64)


def _validate_start(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        name="start",
        required=(
            "schema_version", "message_type", "request_id", "task_id", "operation",
            "runtime_revision", "created_at_unix_ms", "deadline_unix_ms", "model",
            "messages", "sampling", "objective", "memory", "deployment_mode",
        ),
    )
    _identifier(value["request_id"], "request_id")
    _identifier(value["task_id"], "task_id")
    if value["operation"] != "llm.generate":
        raise ContractError("operation must be llm.generate")
    runtime = _string(value["runtime_revision"], "runtime_revision", maximum=64)
    if RUNTIME_REVISION.fullmatch(runtime) is None:
        raise ContractError("runtime_revision must be 64 lowercase hex characters")
    created = _integer(value["created_at_unix_ms"], "created_at_unix_ms", minimum=0)
    deadline = _integer(value["deadline_unix_ms"], "deadline_unix_ms", minimum=1)
    if deadline <= created or deadline - created > 24 * 60 * 60 * 1000:
        raise ContractError("deadline must be after creation and no more than 24 hours later")
    _validate_model(value["model"])

    messages = _array(value["messages"], "messages", minimum=1, maximum=128)
    total_content = 0
    for index, raw_message in enumerate(messages):
        message = _object(raw_message, f"messages[{index}]")
        _exact(message, name=f"messages[{index}]", required=("role", "content"))
        if message["role"] not in {"system", "user", "assistant"}:
            raise ContractError(f"messages[{index}].role is unsupported in v1")
        content = _string(message["content"], f"messages[{index}].content", maximum=65536)
        total_content += len(content)
    if total_content > 524288:
        raise ContractError("combined message content exceeds 524288 characters")

    sampling = _object(value["sampling"], "sampling")
    _exact(sampling, name="sampling", required=("temperature", "top_p", "seed", "stop"))
    _number(sampling["temperature"], "sampling.temperature", minimum=0, maximum=2)
    _number(
        sampling["top_p"], "sampling.top_p", minimum=0, maximum=1,
        exclusive_minimum=True,
    )
    if sampling["seed"] is not None:
        _integer(sampling["seed"], "sampling.seed", minimum=-(2**63), maximum=2**63 - 1)
    stop = _array(sampling["stop"], "sampling.stop", maximum=4)
    normalized_stop = [_string(item, "sampling.stop[]", maximum=128) for item in stop]
    if len(normalized_stop) != len(set(normalized_stop)):
        raise ContractError("sampling.stop must be unique")

    objective = _object(value["objective"], "objective")
    _exact(
        objective,
        name="objective",
        required=("workload_class", "context_tokens", "max_output_tokens", "batch_size"),
        optional=("target_ttft_ms", "min_decode_tokens_per_second"),
    )
    if objective["workload_class"] not in {"interactive", "throughput", "background"}:
        raise ContractError("objective.workload_class is invalid")
    _integer(objective["context_tokens"], "objective.context_tokens", minimum=1, maximum=131072)
    _integer(objective["max_output_tokens"], "objective.max_output_tokens", minimum=1, maximum=8192)
    if objective["batch_size"] != 1 or isinstance(objective["batch_size"], bool):
        raise ContractError("objective.batch_size must be 1 in v1")
    for name in ("target_ttft_ms", "min_decode_tokens_per_second"):
        if name in objective:
            _number(objective[name], f"objective.{name}", minimum=0, exclusive_minimum=True)

    memory = _object(value["memory"], "memory")
    _exact(
        memory,
        name="memory",
        required=("max_runtime_memory_bytes", "ssd_offload"),
        optional=("max_resident_weight_bytes",),
    )
    runtime_bytes = _integer(
        memory["max_runtime_memory_bytes"],
        "memory.max_runtime_memory_bytes",
        minimum=256 * 1024 * 1024,
    )
    if "max_resident_weight_bytes" in memory:
        resident = _integer(
            memory["max_resident_weight_bytes"],
            "memory.max_resident_weight_bytes",
            minimum=1,
        )
        if resident > runtime_bytes:
            raise ContractError("resident weight ceiling cannot exceed runtime memory ceiling")
    if memory["ssd_offload"] != "disabled":
        raise ContractError("ssd_offload must be disabled in v1")
    if value["deployment_mode"] != "supported":
        raise ContractError("deployment_mode must be supported in v1")


def _validate_control(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        name="control",
        required=("schema_version", "message_type", "control_id", "action", "issued_at_unix_ms"),
        optional=("target_request_id",),
    )
    _identifier(value["control_id"], "control_id")
    _integer(value["issued_at_unix_ms"], "issued_at_unix_ms", minimum=0)
    action = value["action"]
    if action not in {"cancel", "drain"}:
        raise ContractError("control.action is invalid")
    if action == "cancel":
        if "target_request_id" not in value:
            raise ContractError("cancel requires target_request_id")
        _identifier(value["target_request_id"], "target_request_id")
    elif "target_request_id" in value:
        raise ContractError("drain must not include target_request_id")


def _validate_event(value: Mapping[str, Any]) -> None:
    optional = (
        "progress", "text_delta", "generated_tokens_delta", "queue_position",
        "queue_depth", "message",
    )
    _exact(
        value,
        name="event",
        required=("schema_version", "message_type", "request_id", "sequence", "kind", "emitted_at_unix_ms"),
        optional=optional,
    )
    _identifier(value["request_id"], "request_id")
    _integer(value["sequence"], "sequence", minimum=0)
    _integer(value["emitted_at_unix_ms"], "emitted_at_unix_ms", minimum=0)
    kind = value["kind"]
    allowed_by_kind = {
        "accepted": {"message"},
        "progress": {"progress", "message"},
        "text_delta": {"text_delta", "generated_tokens_delta"},
        "queue_state": {"queue_position", "queue_depth", "message"},
        "warning": {"message"},
    }
    if kind not in allowed_by_kind:
        raise ContractError("event.kind is invalid")
    present = set(value) & set(optional)
    if not present <= allowed_by_kind[kind]:
        raise ContractError(f"event fields are invalid for kind {kind}")
    required_by_kind = {
        "accepted": set(),
        "progress": {"progress"},
        "text_delta": {"text_delta", "generated_tokens_delta"},
        "queue_state": {"queue_position", "queue_depth"},
        "warning": {"message"},
    }
    if not required_by_kind[kind] <= present:
        raise ContractError(f"event kind {kind} is missing required fields")
    if "progress" in value:
        _number(value["progress"], "progress", minimum=0, maximum=1)
    if "text_delta" in value:
        _string(value["text_delta"], "text_delta", maximum=65536)
    if "generated_tokens_delta" in value:
        _integer(value["generated_tokens_delta"], "generated_tokens_delta", minimum=0, maximum=8192)
    for name in ("queue_position", "queue_depth"):
        if name in value:
            _integer(value[name], name, minimum=0)
    if "message" in value:
        _string(value["message"], "message", maximum=1024)


def _validate_usage(value: Any) -> None:
    usage = _object(value, "usage")
    _exact(usage, name="usage", required=("prompt_tokens", "completion_tokens", "total_tokens"))
    prompt = _integer(usage["prompt_tokens"], "usage.prompt_tokens", minimum=0)
    completion = _integer(usage["completion_tokens"], "usage.completion_tokens", minimum=0)
    total = _integer(usage["total_tokens"], "usage.total_tokens", minimum=0)
    if total != prompt + completion:
        raise ContractError("usage.total_tokens must equal prompt_tokens + completion_tokens")


def _validate_metrics(value: Any) -> None:
    metrics = _object(value, "metrics")
    _exact(
        metrics,
        name="metrics",
        required=(
            "queue_ms", "ttft_ms", "inter_token_latency_ms",
            "decode_tokens_per_second", "peak_memory_bytes", "maximum_batch_size",
        ),
    )
    _number(metrics["queue_ms"], "metrics.queue_ms", minimum=0)
    for name in ("ttft_ms", "inter_token_latency_ms"):
        if metrics[name] is not None:
            _number(metrics[name], f"metrics.{name}", minimum=0)
    if metrics["decode_tokens_per_second"] is not None:
        _number(
            metrics["decode_tokens_per_second"],
            "metrics.decode_tokens_per_second",
            minimum=0,
            exclusive_minimum=True,
        )
    _integer(metrics["peak_memory_bytes"], "metrics.peak_memory_bytes", minimum=0)
    _integer(metrics["maximum_batch_size"], "metrics.maximum_batch_size", minimum=1)


def _validate_execution(value: Any) -> None:
    execution = _object(value, "execution")
    _exact(
        execution,
        name="execution",
        required=("backend", "route", "plan_id", "evidence_ids", "fallback_used"),
        optional=("fallback_reason",),
    )
    if execution["backend"] not in {"mlx", "metal", "coreml", "ane", "cpu", "hybrid"}:
        raise ContractError("execution.backend is invalid")
    if execution["route"] not in {"batch_generator", "direct_single_sequence", "native_phase_program"}:
        raise ContractError("execution.route is invalid")
    _identifier(execution["plan_id"], "execution.plan_id")
    evidence = _array(execution["evidence_ids"], "execution.evidence_ids", maximum=16)
    normalized = [_identifier(item, "execution.evidence_ids[]") for item in evidence]
    if len(normalized) != len(set(normalized)):
        raise ContractError("execution.evidence_ids must be unique")
    if not isinstance(execution["fallback_used"], bool):
        raise ContractError("execution.fallback_used must be a boolean")
    if execution["fallback_used"]:
        if "fallback_reason" not in execution:
            raise ContractError("fallback_reason is required when fallback_used is true")
        _string(execution["fallback_reason"], "execution.fallback_reason", maximum=512)
    elif "fallback_reason" in execution:
        raise ContractError("fallback_reason is forbidden when fallback_used is false")


def _validate_failure(value: Any) -> Mapping[str, Any]:
    failure = _object(value, "failure")
    _exact(failure, name="failure", required=("code", "stage", "retryable", "message"))
    _identifier(failure["code"], "failure.code")
    if failure["stage"] not in {"not_started", "running", "partial_output"}:
        raise ContractError("failure.stage is invalid")
    if not isinstance(failure["retryable"], bool):
        raise ContractError("failure.retryable must be a boolean")
    _string(failure["message"], "failure.message", maximum=1024)
    return failure


def _validate_receipt(value: Mapping[str, Any]) -> None:
    _exact(
        value,
        name="receipt",
        required=(
            "schema_version", "message_type", "request_id", "terminal_sequence",
            "status", "finish_reason", "completed_at_unix_ms", "usage", "metrics",
            "execution",
        ),
        optional=("output_text", "failure"),
    )
    _identifier(value["request_id"], "request_id")
    _integer(value["terminal_sequence"], "terminal_sequence", minimum=0)
    _integer(value["completed_at_unix_ms"], "completed_at_unix_ms", minimum=0)
    status = value["status"]
    finish = value["finish_reason"]
    allowed_finish = {
        "completed": {"stop", "length"},
        "failed": {"error"},
        "cancelled": {"cancelled"},
        "rejected": {"rejected"},
    }
    if status not in allowed_finish or finish not in allowed_finish[status]:
        raise ContractError("receipt status and finish_reason are inconsistent")
    if status == "completed":
        if "output_text" not in value or "failure" in value:
            raise ContractError("completed receipt requires output_text and forbids failure")
    else:
        if "failure" not in value:
            raise ContractError("non-completed receipt requires failure")
        failure = _validate_failure(value["failure"])
        if status == "rejected" and failure["stage"] != "not_started":
            raise ContractError("rejected receipt failure stage must be not_started")
    if "output_text" in value:
        _string(value["output_text"], "output_text", minimum=0, maximum=1048576)
    _validate_usage(value["usage"])
    _validate_metrics(value["metrics"])
    _validate_execution(value["execution"])


def validate_document(value: Mapping[str, Any], raw: bytes) -> str:
    if value.get("schema_version") != SCHEMA_VERSION or isinstance(value.get("schema_version"), bool):
        raise ContractError("schema_version must be integer 1")
    message_type = value.get("message_type")
    if message_type not in MESSAGE_LIMITS:
        raise ContractError("message_type is invalid")
    if len(raw) > MESSAGE_LIMITS[message_type]:
        raise ContractError(f"{message_type} exceeds its encoded byte limit")
    validators = {
        "start": _validate_start,
        "control": _validate_control,
        "event": _validate_event,
        "receipt": _validate_receipt,
    }
    validators[message_type](value)
    return message_type


def verify_fixture_tree(root: Path = KIT_ROOT) -> dict[str, int]:
    schema, _ = load_document(root / "schemas/strata-wire-v1.schema.json")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ContractError("schema must declare JSON Schema draft 2020-12")
    counts = {"valid": 0, "invalid": 0}
    for path in sorted((root / "fixtures/valid").glob("*.json")):
        value, raw = load_document(path)
        validate_document(value, raw)
        counts["valid"] += 1
    for path in sorted((root / "fixtures/invalid").glob("*.json")):
        try:
            value, raw = load_document(path)
            validate_document(value, raw)
        except ContractError:
            counts["invalid"] += 1
        else:
            raise ContractError(f"invalid fixture unexpectedly passed: {path.name}")
    if counts["valid"] == 0 or counts["invalid"] == 0:
        raise ContractError("fixture tree must contain valid and invalid documents")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=KIT_ROOT)
    args = parser.parse_args()
    try:
        counts = verify_fixture_tree(args.root.resolve())
    except (OSError, ContractError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"success": True, **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

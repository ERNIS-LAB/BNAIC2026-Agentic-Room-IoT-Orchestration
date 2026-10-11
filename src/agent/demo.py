"""Live gateway demo that exercises every registered agent tool."""

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx

from agent.app import DEFAULT_TOOL_NAMES
from gateway.config import settings as default_settings


@dataclass(frozen=True)
class DemoScenario:
    name: str
    prompt: str
    expected_tools: tuple[str, ...]


@dataclass
class ScenarioResult:
    name: str
    thread_id: str
    elapsed_s: float
    observed_tools: set[str] = field(default_factory=set)
    final_answer: str = ""
    final_payload: dict[str, Any] | None = None


DEMO_SCENARIOS = (
    DemoScenario(
        name="Camera check",
        prompt=(
            "Look at the current camera frame and describe whether the room appears "
            "occupied. Use read_sensors with camera before answering."
        ),
        expected_tools=DEFAULT_TOOL_NAMES,
    ),
    DemoScenario(
        name="Temperature check",
        prompt=(
            "Check the current room temperature and tell me if it is comfortable "
            "for a meeting. Use read_sensors with temperature before answering."
        ),
        expected_tools=DEFAULT_TOOL_NAMES,
    ),
    DemoScenario(
        name="Light and sound check",
        prompt=(
            "What are the current light and sound levels in the room? "
            "Use read_sensors before answering."
        ),
        expected_tools=DEFAULT_TOOL_NAMES,
    ),
    DemoScenario(
        name="Full readiness check",
        prompt=(
            "I need this room for a meeting right now. Check all sensors "
            "(camera, temperature, light, sound) and give a recommendation."
        ),
        expected_tools=DEFAULT_TOOL_NAMES,
    ),
)


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a live multimodal-agent demo and verify registered tool coverage.",
    )
    parser.add_argument(
        "--base-url",
        default=default_settings.gateway_base_url,
        help="Gateway base URL, for example http://localhost:8000 or http://<hub-ip>:8000.",
    )
    parser.add_argument(
        "--thread-prefix",
        default=None,
        help="Optional thread-id prefix. Defaults to a random demo prefix.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=default_settings.request_timeout,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Print raw final JSON payloads after each scenario.",
    )
    return parser


def _iter_agent_events(base_url: str, query: str, thread_id: str, timeout: float):
    stream_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0), read=None)
    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/v1/agent/stream",
        json={
            "thread_id": thread_id,
            "user_input": query,
            "prior_messages": [],
            "include_trace": True,
        },
        timeout=stream_timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                yield json.loads(line)


def _usage_values(payload: dict[str, Any]) -> tuple[int, int, int]:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    tokens_in = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    tokens_out = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total = int(usage.get("total_tokens", tokens_in + tokens_out) or 0)
    return tokens_in, tokens_out, total


def _step_seconds(payload: dict[str, Any]) -> float:
    return float(payload.get("duration_ms") or 0.0) / 1000.0


def _tool_names_from_text(text: str) -> set[str]:
    return {tool_name for tool_name in DEFAULT_TOOL_NAMES if tool_name in text}


def observed_tools_from_payload(payload: dict[str, Any]) -> set[str]:
    observed: set[str] = set()
    name = str(payload.get("name") or "")
    summary = str(payload.get("summary") or "")
    observed.update(_tool_names_from_text(name))
    observed.update(_tool_names_from_text(summary))

    for message in payload.get("messages") or []:
        if isinstance(message, dict):
            observed.update(_tool_names_from_text(str(message.get("content") or "")))
    return observed


def observed_tools_from_final(result: dict[str, Any]) -> set[str]:
    observed: set[str] = set()
    for observation in result.get("tool_observations") or []:
        if isinstance(observation, dict):
            tool_name = str(observation.get("tool_name") or "")
            if tool_name:
                observed.add(tool_name)

    run_log = result.get("run_log") if isinstance(result.get("run_log"), dict) else {}
    for step in run_log.get("steps") or []:
        if isinstance(step, dict):
            observed.update(_tool_names_from_text(str(step.get("name") or "")))
            observed.update(_tool_names_from_text(str(step.get("summary") or "")))

    for event in result.get("trace") or []:
        if isinstance(event, dict):
            payload = event.get("payload")
            if isinstance(payload, dict):
                observed.update(observed_tools_from_payload(payload))

    return observed


def coverage_missing(observed_tools: Iterable[str]) -> list[str]:
    observed = set(observed_tools)
    return [tool_name for tool_name in DEFAULT_TOOL_NAMES if tool_name not in observed]


def _format_token_line(payload: dict[str, Any], elapsed_s: float) -> str:
    tokens_in, tokens_out, total = _usage_values(payload)
    if not total:
        return ""
    tok_per_s = tokens_out / elapsed_s if elapsed_s > 0 and tokens_out else 0.0
    return (
        f"tokens in={tokens_in}, out={tokens_out}, total={total}, {tok_per_s:.1f} tok/s"
    )


def _print_header(base_url: str, scenario_count: int) -> None:
    print("Multimodal Agent Demo", flush=True)
    print("=====================", flush=True)
    print(f"Gateway: {base_url}", flush=True)
    print(f"Registered tools: {', '.join(DEFAULT_TOOL_NAMES)}", flush=True)
    print(f"Scenarios: {scenario_count}", flush=True)
    print(flush=True)


def _print_scenario_header(index: int, scenario: DemoScenario, thread_id: str) -> None:
    print(f"Scenario {index}: {scenario.name}", flush=True)
    print("-" * (11 + len(str(index)) + len(scenario.name)), flush=True)
    print(f"Thread: {thread_id}", flush=True)
    print(f"Expected tools: {', '.join(scenario.expected_tools)}", flush=True)
    print("User:", flush=True)
    print(f"  {scenario.prompt}", flush=True)
    print(flush=True)


def _print_trace_event(payload: dict[str, Any], observed: set[str]) -> None:
    sequence = int(payload.get("sequence") or 0)
    stage = str(payload.get("stage") or "")
    name = str(payload.get("name") or stage or "step")
    summary = str(payload.get("summary") or "")
    elapsed_s = _step_seconds(payload)
    token_line = _format_token_line(payload, elapsed_s)

    if stage == "model" and name.startswith("tool_call:"):
        tools = sorted(observed_tools_from_payload(payload))
        observed.update(tools)
        print(
            f"  Step {sequence}: model is calling tools ({elapsed_s:.2f}s)", flush=True
        )
        print(
            f"    Tools: {', '.join(tools) if tools else name.removeprefix('tool_call:')}",
            flush=True,
        )
    elif stage == "model":
        print(
            f"  Step {sequence}: model is reasoning/responding ({elapsed_s:.2f}s)",
            flush=True,
        )
        if summary:
            print(f"    {summary}", flush=True)
    elif stage == "tools":
        tools = sorted(observed_tools_from_payload(payload))
        observed.update(tools)
        print(f"  Step {sequence}: tool response ({elapsed_s:.2f}s)", flush=True)
        print(f"    Tools: {', '.join(tools) if tools else name}", flush=True)
        if summary:
            print(f"    Output: {summary}", flush=True)
    else:
        print(f"  Step {sequence}: {stage or name} ({elapsed_s:.2f}s)", flush=True)
        if summary:
            print(f"    {summary}", flush=True)

    if token_line:
        print(f"    {token_line}", flush=True)


def _print_run_timing(result: dict[str, Any]) -> None:
    run_log = result.get("run_log") if isinstance(result.get("run_log"), dict) else None
    if not run_log:
        return

    print("  Timing detail:", flush=True)
    for step in run_log.get("steps") or []:
        if not isinstance(step, dict):
            continue
        duration_s = float(step.get("duration_ms") or 0.0) / 1000.0
        usage = step.get("usage") if isinstance(step.get("usage"), dict) else {}
        tokens_in = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        tokens_out = int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        token_text = (
            f" tokens={tokens_in}/{tokens_out}" if tokens_in or tokens_out else ""
        )
        print(
            f"    #{int(step.get('sequence') or 0):02d} "
            f"{str(step.get('stage') or ''):<7} "
            f"{str(step.get('name') or '')[:36]:<36} "
            f"{duration_s:>6.2f}s{token_text}",
            flush=True,
        )


def run_scenario(
    base_url: str,
    scenario: DemoScenario,
    index: int,
    thread_prefix: str,
    timeout: float,
    *,
    raw_json: bool = False,
) -> ScenarioResult:
    thread_id = f"{thread_prefix}-{index}"
    _print_scenario_header(index, scenario, thread_id)

    observed: set[str] = set()
    final_payload: dict[str, Any] | None = None
    final_answer = ""
    token_stream_open = False
    started = time.perf_counter()

    try:
        _log("Submitting scenario to the gateway stream.")
        for event in _iter_agent_events(base_url, scenario.prompt, thread_id, timeout):
            event_type = str(event.get("event") or "")
            payload = (
                event.get("payload") if isinstance(event.get("payload"), dict) else {}
            )

            if event_type == "trace":
                if token_stream_open:
                    print(flush=True)
                    token_stream_open = False
                payload = dict(payload)
                payload.setdefault("stage", str(event.get("stage") or ""))
                _print_trace_event(payload, observed)
            elif event_type == "token":
                text = payload.get("text")
                if isinstance(text, str) and text:
                    if not token_stream_open:
                        print("  Final response stream:", flush=True)
                        print("    ", end="", flush=True)
                        token_stream_open = True
                    print(text, end="", flush=True)
            elif event_type == "final":
                if token_stream_open:
                    print(flush=True)
                    token_stream_open = False
                final_payload = (
                    event.get("result") if isinstance(event.get("result"), dict) else {}
                )
                observed.update(observed_tools_from_final(final_payload))
                final_answer = str(final_payload.get("final_answer") or "")
                print("  Final answer:", flush=True)
                print(f"    {final_answer}", flush=True)
                _print_run_timing(final_payload)
                if raw_json:
                    print("  Raw final payload:", flush=True)
                    print(json.dumps(final_payload, indent=2), flush=True)
                break
            elif event_type == "error":
                detail = event.get("detail")
                raise RuntimeError(
                    json.dumps(detail) if isinstance(detail, dict) else str(event)
                )
    except httpx.HTTPStatusError as exc:
        print(
            f"Gateway returned {exc.response.status_code} for {exc.request.url}",
            flush=True,
        )
        print(exc.response.text, flush=True)
        raise
    except httpx.ReadTimeout:
        print(
            "The gateway stream timed out while waiting for the next event.", flush=True
        )
        raise
    except httpx.HTTPError as exc:
        print(f"Could not reach the gateway at {base_url}: {exc}", flush=True)
        raise

    elapsed_s = time.perf_counter() - started
    missing_expected = [
        tool for tool in scenario.expected_tools if tool not in observed
    ]
    if missing_expected:
        print(f"  Scenario coverage: missing {', '.join(missing_expected)}", flush=True)
    else:
        print("  Scenario coverage: passed", flush=True)
    print(f"  Scenario time: {elapsed_s:.2f}s", flush=True)
    print(flush=True)

    return ScenarioResult(
        name=scenario.name,
        thread_id=thread_id,
        elapsed_s=elapsed_s,
        observed_tools=observed,
        final_answer=final_answer,
        final_payload=final_payload,
    )


def _print_coverage_summary(
    results: list[ScenarioResult], total_elapsed_s: float
) -> bool:
    observed: set[str] = set()
    for result in results:
        observed.update(result.observed_tools)
    missing = coverage_missing(observed)

    print("Tool Coverage", flush=True)
    print("-------------", flush=True)
    for tool_name in DEFAULT_TOOL_NAMES:
        status = "PASS" if tool_name in observed else "MISS"
        scenarios = [
            result.name for result in results if tool_name in result.observed_tools
        ]
        scenario_text = ", ".join(scenarios) if scenarios else "-"
        print(f"  {status:<4} {tool_name:<28} {scenario_text}", flush=True)
    print(flush=True)
    print("Timing Summary", flush=True)
    print("--------------", flush=True)
    for result in results:
        print(f"  {result.name:<24} {result.elapsed_s:>7.2f}s", flush=True)
    print(f"  {'total':<24} {total_elapsed_s:>7.2f}s", flush=True)
    print(flush=True)

    if missing:
        print(f"Missing registered tools: {', '.join(missing)}", flush=True)
        return False
    print("All registered tools were exercised.", flush=True)
    return True


def main() -> int:
    args = build_parser().parse_args()
    thread_prefix = args.thread_prefix or f"demo-{uuid.uuid4().hex[:8]}"
    _print_header(args.base_url, len(DEMO_SCENARIOS))

    results: list[ScenarioResult] = []
    started = time.perf_counter()
    for index, scenario in enumerate(DEMO_SCENARIOS, start=1):
        try:
            results.append(
                run_scenario(
                    base_url=args.base_url,
                    scenario=scenario,
                    index=index,
                    thread_prefix=thread_prefix,
                    timeout=args.timeout,
                    raw_json=args.raw_json,
                )
            )
        except Exception:
            return 1

    total_elapsed_s = time.perf_counter() - started
    return 0 if _print_coverage_summary(results, total_elapsed_s) else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from dataclasses import dataclass
from typing import Any

from sample_data import build_sample_cases
from signature_baseline import SupportTicketTriage as BaselineSupportTicketTriage
from signature_typesafe import SupportTicketTriage as TypesafeSupportTicketTriage

import dspy
from typesafe_dspy import (
    compare_predictions,
    configure_typesafe,
    plan_signature,
    render_prediction_comparison,
    typesafe_results,
    typesafe_timings,
)

LUNA_MODEL = "gpt-5.6-luna"
LUNA_INPUT_USD_PER_MILLION = 0.20
LUNA_OUTPUT_USD_PER_MILLION = 1.20
TYPESAFE_INPUT_USD_PER_MILLION = 0.042


@dataclass(frozen=True)
class CaseTiming:
    baseline_seconds: float
    typesafe_seconds: float
    hybrid_dspy_seconds: float
    hybrid_total_seconds: float


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class CaseCost:
    baseline_luna: TokenUsage
    typesafe: TokenUsage
    hybrid_luna: TokenUsage

    @property
    def baseline_usd(self) -> float:
        return _luna_cost(self.baseline_luna)

    @property
    def hybrid_usd(self) -> float:
        return _luna_cost(self.hybrid_luna) + (
            self.typesafe.input_tokens * TYPESAFE_INPUT_USD_PER_MILLION / 1_000_000
        )


class UsageRecordingTypesafeClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.last_usage = TokenUsage()

    def system_one(self, *args, **kwargs):
        response = self._client.system_one(*args, **kwargs)
        if response.usage.input_tokens is None or response.usage.output_tokens is None:
            raise RuntimeError("Typesafe did not report complete token usage; cannot calculate benchmark cost.")
        self.last_usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response


def build_ticket_document(signature: type[dspy.Signature], inputs: dict) -> dict:
    return {
        "task": signature.instructions,
        "ticket": inputs["ticket"],
        "account": inputs["account"],
        "signals": inputs["signals"],
        "history": inputs["history"],
        "rubric": {
            "customer_impacting": ("Use current operational evidence rather than customer frustration alone."),
            "needs_human_now": ("Prefer escalation for active high-value impact or ambiguous safety-critical cases."),
            "owner_team": (
                "Route to the team that should take first response, not necessarily the eventual root cause."
            ),
            "severity_band": "Use low/medium/high/critical for operational urgency.",
            "severity_score": "Estimate operational severity on a 1 to 5 scale.",
        },
    }


def configure_runtime() -> UsageRecordingTypesafeClient:
    from typesafe_sdk import TypeSafeClient

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    typesafe_api_key = os.environ.get("TYPESAFE_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set.")
    if not typesafe_api_key:
        raise RuntimeError("TYPESAFE_API_KEY must be set.")

    openai_model = os.environ.get("OPENAI_MODEL", LUNA_MODEL)
    typesafe_model = os.environ.get("TYPESAFE_MODEL", "speed_latest")
    if openai_model != LUNA_MODEL:
        raise RuntimeError(
            f"This fixed-price benchmark requires OPENAI_MODEL={LUNA_MODEL}; got {openai_model}."
        )

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    dspy.configure(
        lm=dspy.LM(
            f"openai/{openai_model}",
            cache=False,
        ),
        callbacks=[],
        track_usage=True,
    )
    typesafe_client = UsageRecordingTypesafeClient(TypeSafeClient(api_key=typesafe_api_key))
    configure_typesafe(
        client=typesafe_client,
        model=typesafe_model,
        document_builder=build_ticket_document,
    )
    return typesafe_client


def signature_snapshot(signature: type[dspy.Signature]) -> dict[str, Any]:
    return {
        "instructions": signature.instructions,
        "inputs": [_field_snapshot(name, field) for name, field in signature.input_fields.items()],
        "outputs": [_field_snapshot(name, field) for name, field in signature.output_fields.items()],
    }


def _field_snapshot(name: str, field: Any) -> dict[str, Any]:
    extras = field.json_schema_extra or {}
    return {
        "name": name,
        "annotation": repr(field.annotation),
        "desc": extras.get("desc"),
        "constraints": extras.get("constraints"),
    }


def ensure_signature_parity(
    baseline: type[dspy.Signature],
    candidate: type[dspy.Signature],
) -> None:
    baseline_snapshot = signature_snapshot(baseline)
    candidate_snapshot = signature_snapshot(candidate)
    if baseline_snapshot != candidate_snapshot:
        raise RuntimeError(
            "Baseline and Typesafe signatures are not identical.\n"
            f"baseline={json.dumps(baseline_snapshot, indent=2, sort_keys=True)}\n"
            f"typesafe={json.dumps(candidate_snapshot, indent=2, sort_keys=True)}"
        )


def render_case_context(case: dict) -> str:
    ticket = case["ticket"]
    signals = case["signals"]
    parts = [
        f"tier={ticket['customer_tier']}",
        f"region={ticket['region']}",
        f"channel={ticket['channel']}",
        f"error_rate_5m={signals['error_rate_5m']:.2f}",
        f"latency_p95_ms={signals['latency_p95_ms']}",
        f"affected_users={signals['affected_users_estimate']}",
    ]
    return "  ".join(parts)


def non_text_output_fields(signature: type[dspy.Signature]) -> list[str]:
    return [name for name, field in signature.output_fields.items() if field.annotation is not str]


def text_output_fields(signature: type[dspy.Signature]) -> list[str]:
    return [name for name, field in signature.output_fields.items() if field.annotation is str]


def render_prediction_summary(
    label: str,
    signature: type[dspy.Signature],
    result,
) -> str:
    typesafe_meta = typesafe_results(result)
    parts = [
        f"{field_name}={format_output_value(getattr(result, field_name))}"
        for field_name in non_text_output_fields(signature)
    ]
    for field_name, meta in typesafe_meta.items():
        if meta.probability is not None:
            parts.append(f"{field_name}_probability={meta.probability:.2f}")
            continue
        if meta.probabilities:
            probability = meta.probabilities.get(meta.value)
            if probability is not None:
                parts.append(f"{field_name}_confidence={probability:.2f}")
    return f"  {label:<8}  " + "  ".join(parts)


def render_timing_line(timing: CaseTiming) -> str:
    speedup = 0.0
    if timing.baseline_seconds > 0:
        speedup = 1.0 - (timing.hybrid_total_seconds / timing.baseline_seconds)
    return "  ".join(
        [
            f"baseline_total={timing.baseline_seconds:.3f}s",
            f"typesafe_eval={timing.typesafe_seconds:.3f}s",
            f"hybrid_dspy={timing.hybrid_dspy_seconds:.3f}s",
            f"hybrid_total={timing.hybrid_total_seconds:.3f}s",
            f"speedup={speedup:+.1%}",
        ]
    )


def prediction_usage(result) -> TokenUsage:
    usage_by_model = result.get_lm_usage() or {}
    if not usage_by_model or any(
        "prompt_tokens" not in usage or "completion_tokens" not in usage for usage in usage_by_model.values()
    ):
        raise RuntimeError("Luna did not report complete token usage; cannot calculate benchmark cost.")
    return TokenUsage(
        input_tokens=sum(usage.get("prompt_tokens", 0) for usage in usage_by_model.values()),
        output_tokens=sum(usage.get("completion_tokens", 0) for usage in usage_by_model.values()),
    )


def _luna_cost(usage: TokenUsage) -> float:
    return (
        usage.input_tokens * LUNA_INPUT_USD_PER_MILLION
        + usage.output_tokens * LUNA_OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def render_cost_line(cost: CaseCost) -> str:
    savings = 0.0
    if cost.baseline_usd > 0:
        savings = 1.0 - (cost.hybrid_usd / cost.baseline_usd)
    return "  ".join(
        [
            f"baseline_luna_input={cost.baseline_luna.input_tokens}",
            f"baseline_luna_output={cost.baseline_luna.output_tokens}",
            f"typesafe_input={cost.typesafe.input_tokens}",
            f"typesafe_output={cost.typesafe.output_tokens}",
            f"hybrid_luna_input={cost.hybrid_luna.input_tokens}",
            f"hybrid_luna_output={cost.hybrid_luna.output_tokens}",
            f"baseline_cost=${cost.baseline_usd:.6f}",
            f"hybrid_cost=${cost.hybrid_usd:.6f}",
            f"savings={savings:+.1%}",
        ]
    )


def format_output_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_text_pair(field_name: str, baseline, candidate) -> str:
    title = f"  {field_name}"
    baseline_text = _wrap_text("baseline", getattr(baseline, field_name))
    candidate_text = _wrap_text("typesafe", getattr(candidate, field_name))
    return "\n".join([title, baseline_text, candidate_text])


def _wrap_text(label: str, value: str, *, width: int = 100) -> str:
    prefix = f"    {label:<8}  "
    return textwrap.fill(
        value,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--show-document", action="store_true")
    args = parser.parse_args(argv)

    ensure_signature_parity(
        BaselineSupportTicketTriage,
        TypesafeSupportTicketTriage,
    )

    cases = build_sample_cases()[: args.limit]
    typesafe_client = configure_runtime()

    plan = plan_signature(TypesafeSupportTicketTriage)
    print("Signature plan")
    print("  signature_parity=exact")
    print(f"  typesafe_fields={', '.join(plan.typesafe_field_names)}")
    print(f"  residual_dspy_fields={', '.join(plan.remaining_output_names)}")
    print("  dspy_cache=off  lm_cache=off")

    if args.show_document and cases:
        preview = build_ticket_document(
            TypesafeSupportTicketTriage,
            {
                "ticket": cases[0]["ticket"],
                "account": cases[0]["account"],
                "signals": cases[0]["signals"],
                "history": cases[0]["history"],
            },
        )
        print("\nTypesafe document preview")
        print(json.dumps(preview, indent=2))

    baseline_predictor = dspy.Predict(BaselineSupportTicketTriage)
    typesafe_predictor = dspy.Predict(TypesafeSupportTicketTriage)
    timings: list[CaseTiming] = []
    costs: list[CaseCost] = []
    long_text_fields = text_output_fields(BaselineSupportTicketTriage)

    for index, case in enumerate(cases, start=1):
        inputs = {
            "ticket": case["ticket"],
            "account": case["account"],
            "signals": case["signals"],
            "history": case["history"],
        }

        baseline_start = time.perf_counter()
        baseline_result = baseline_predictor(**inputs)
        baseline_seconds = time.perf_counter() - baseline_start
        baseline_usage = prediction_usage(baseline_result)

        typesafe_result = typesafe_predictor(**inputs)
        split_timing = typesafe_timings(typesafe_result)
        assert split_timing is not None

        timing = CaseTiming(
            baseline_seconds=baseline_seconds,
            typesafe_seconds=split_timing.typesafe_seconds,
            hybrid_dspy_seconds=split_timing.dspy_seconds,
            hybrid_total_seconds=split_timing.total_seconds,
        )
        timings.append(timing)
        cost = CaseCost(
            baseline_luna=baseline_usage,
            typesafe=typesafe_client.last_usage,
            hybrid_luna=prediction_usage(typesafe_result),
        )
        costs.append(cost)

        comparison = compare_predictions(
            BaselineSupportTicketTriage,
            baseline_result,
            typesafe_result,
            baseline_label="baseline",
            candidate_label="typesafe",
        )
        changed_fields = ", ".join(comparison.changed_field_names) or "none"

        print(f"\nCase {index}: {case['ticket']['title']}")
        print(f"  context    {render_case_context(case)}")
        print(
            render_prediction_summary(
                "baseline",
                BaselineSupportTicketTriage,
                baseline_result,
            )
        )
        print(
            render_prediction_summary(
                "typesafe",
                TypesafeSupportTicketTriage,
                typesafe_result,
            )
        )
        print(f"  changed_fields={changed_fields}")
        print(render_prediction_comparison(comparison, max_value_width=28))
        print(f"  timings    {render_timing_line(timing)}")
        print(f"  usage      {render_cost_line(cost)}")
        for field_name in long_text_fields:
            print(render_text_pair(field_name, baseline_result, typesafe_result))

    if timings:
        avg_baseline = sum(item.baseline_seconds for item in timings) / len(timings)
        avg_typesafe = sum(item.typesafe_seconds for item in timings) / len(timings)
        avg_hybrid_dspy = sum(item.hybrid_dspy_seconds for item in timings) / len(timings)
        avg_hybrid_total = sum(item.hybrid_total_seconds for item in timings) / len(timings)
        batch_speedup = 0.0
        if avg_baseline > 0:
            batch_speedup = 1.0 - (avg_hybrid_total / avg_baseline)

        print("\nBatch timing summary")
        print(
            "  ".join(
                [
                    f"cases={len(timings)}",
                    f"avg_baseline_total={avg_baseline:.3f}s",
                    f"avg_typesafe_eval={avg_typesafe:.3f}s",
                    f"avg_hybrid_dspy={avg_hybrid_dspy:.3f}s",
                    f"avg_hybrid_total={avg_hybrid_total:.3f}s",
                    f"avg_speedup={batch_speedup:+.1%}",
                ]
            )
        )

        total_baseline_cost = sum(item.baseline_usd for item in costs)
        total_hybrid_cost = sum(item.hybrid_usd for item in costs)
        cost_savings = 0.0
        if total_baseline_cost > 0:
            cost_savings = 1.0 - (total_hybrid_cost / total_baseline_cost)

        print("\nBatch cost summary")
        print(
            "  ".join(
                [
                    f"cases={len(costs)}",
                    f"total_baseline_cost=${total_baseline_cost:.6f}",
                    f"total_hybrid_cost=${total_hybrid_cost:.6f}",
                    f"avg_baseline_cost=${total_baseline_cost / len(costs):.6f}",
                    f"avg_hybrid_cost=${total_hybrid_cost / len(costs):.6f}",
                    f"cost_savings={cost_savings:+.1%}",
                    "rates=luna_input_0.20,luna_output_1.20,typesafe_input_0.042_per_1M",
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


@dataclass(frozen=True)
class CaseTiming:
    baseline_seconds: float
    typesafe_seconds: float
    hybrid_dspy_seconds: float
    hybrid_total_seconds: float


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


def configure_runtime() -> None:
    from typesafe_sdk import TypeSafeClient

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    typesafe_api_key = os.environ.get("TYPESAFE_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set.")
    if not typesafe_api_key:
        raise RuntimeError("TYPESAFE_API_KEY must be set.")

    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    typesafe_model = os.environ.get("TYPESAFE_MODEL", "speed_latest")

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    dspy.configure(
        lm=dspy.LM(
            f"openai/{openai_model}",
            cache=False,
        ),
        callbacks=[],
        track_usage=False,
    )
    configure_typesafe(
        client=TypeSafeClient(api_key=typesafe_api_key),
        model=typesafe_model,
        document_builder=build_ticket_document,
    )


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
    configure_runtime()

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

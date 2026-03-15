from __future__ import annotations

from typing import Literal

import dspy


class SupportTicketTriage(dspy.Signature):
    """
    Triage a support ticket using typed routing decisions and drafted text
    outputs.

    Keep freeform string outputs concise.
    Use plain text sentences only. Do not use bullet lists.
    """

    ticket: dict = dspy.InputField(desc="Ticket metadata and freeform customer issue summary.")
    account: dict = dspy.InputField(desc="Customer account context, including revenue and renewal pressure.")
    signals: dict = dspy.InputField(desc="Operational telemetry and recent impact indicators.")
    history: list[dict] = dspy.InputField(desc="Related incidents or recent changes that may explain the issue.")

    customer_impacting: bool = dspy.OutputField(desc="Whether the issue is actively affecting customers now.")
    needs_human_now: bool = dspy.OutputField(desc="Whether the issue should be escalated to a human immediately.")
    owner_team: Literal[
        "api-platform",
        "billing",
        "checkout",
        "infra",
        "support-ops",
    ] = dspy.OutputField(desc="The team that should own the first response.")
    severity_band: Literal["low", "medium", "high", "critical"] = dspy.OutputField(
        desc="A coarse severity bucket for operational routing."
    )
    severity_score: float = dspy.OutputField(desc="A severity score from 1 to 5.")
    internal_summary: str = dspy.OutputField(desc="A concise internal triage note under 20 words.")

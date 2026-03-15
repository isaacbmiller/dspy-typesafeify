from __future__ import annotations


def build_sample_cases() -> list[dict]:
    return [
        {
            "ticket": {
                "title": "Enterprise checkout timeouts during peak traffic",
                "body": (
                    "Customers report intermittent checkout failures between 9:10 and 9:25 AM PST. "
                    "Three enterprise accounts escalated through Slack and p95 latency reached 4.2s."
                ),
                "channel": "slack-escalation",
                "region": "us-east-1",
                "customer_tier": "enterprise",
            },
            "account": {
                "name": "Northwind Health",
                "contract_value_usd": 420000,
                "renewal_days": 18,
                "open_escalations_30d": 2,
            },
            "signals": {
                "error_rate_5m": 0.09,
                "latency_p95_ms": 4200,
                "affected_users_estimate": 16000,
                "payment_provider_errors": 2,
            },
            "history": [
                {
                    "incident": "INC-2041",
                    "resolution": "Increased checkout worker pool.",
                    "days_ago": 11,
                },
                {
                    "incident": "INC-1987",
                    "resolution": "Reverted session cache regression.",
                    "days_ago": 37,
                },
            ],
        },
        {
            "ticket": {
                "title": "Invoices missing VAT labels for EU customers",
                "body": (
                    "Several finance admins noticed PDF invoices missing VAT display text after the most recent "
                    "billing template rollout. Charges appear correct."
                ),
                "channel": "zendesk",
                "region": "eu-west-1",
                "customer_tier": "mid-market",
            },
            "account": {
                "name": "Atlas Retail",
                "contract_value_usd": 74000,
                "renewal_days": 104,
                "open_escalations_30d": 0,
            },
            "signals": {
                "error_rate_5m": 0.0,
                "latency_p95_ms": 310,
                "affected_users_estimate": 41,
                "payment_provider_errors": 0,
            },
            "history": [
                {
                    "incident": "BILL-884",
                    "resolution": "Restored invoice locale mapping.",
                    "days_ago": 63,
                }
            ],
        },
        {
            "ticket": {
                "title": "Webhook retries spiking after regional network flap",
                "body": (
                    "Multiple integrations retried successfully after a short us-west-2 network event. "
                    "Customers are asking whether retries indicate data loss."
                ),
                "channel": "email",
                "region": "us-west-2",
                "customer_tier": "self-serve",
            },
            "account": {
                "name": "Various self-serve accounts",
                "contract_value_usd": 0,
                "renewal_days": None,
                "open_escalations_30d": 0,
            },
            "signals": {
                "error_rate_5m": 0.02,
                "latency_p95_ms": 870,
                "affected_users_estimate": 930,
                "payment_provider_errors": 0,
            },
            "history": [
                {
                    "incident": "NET-771",
                    "resolution": "Traffic failed over to alternate path.",
                    "days_ago": 5,
                }
            ],
        },
    ]

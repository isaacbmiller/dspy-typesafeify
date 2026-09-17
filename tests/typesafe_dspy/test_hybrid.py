import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from typesafe_dspy import (
    PredictionComparison,
    TypesafeConfig,
    TypesafeFieldConfig,
    TypesafePredict,
    TypesafeTiming,
    compare_predictions,
    configure_typesafe,
    disable_typesafe,
    enable_typesafe,
    is_typesafeified,
    plan_signature,
    render_prediction_comparison,
    typesafe_results,
    typesafe_signature,
    typesafe_timings,
    typesafeify,
)


@pytest.fixture(autouse=True)
def reset_dspy_settings():
    dspy.configure(lm=None, adapter=None, callbacks=[], track_usage=False)
    disable_typesafe()
    yield
    disable_typesafe()
    dspy.configure(lm=None, adapter=None, callbacks=[], track_usage=False)


@dataclass(frozen=True)
class FakePrompt:
    kind: str
    instructions: object
    options: dict[str, object] | None = None
    levels: dict[int, object] | None = None


class FakePromptFactory:
    def create_noul(self, *, instructions):
        return FakePrompt(kind="noul", instructions=instructions)

    def create_choice(self, *, instructions, options):
        return FakePrompt(kind="choice", instructions=instructions, options=dict(options))

    def create_score(self, *, instructions, levels):
        return FakePrompt(kind="score", instructions=instructions, levels=dict(levels))


class FakeEvaluation:
    def __init__(self, *, nouls=None, choices=None, scores=None):
        self.answers = {
            **{key: SimpleNamespace(noul=value) for key, value in (nouls or {}).items()},
            **{
                key: SimpleNamespace(
                    choice=payload["choice"],
                    probabilities=payload["probabilities"],
                    confidence=payload.get("confidence"),
                )
                for key, payload in (choices or {}).items()
            },
            **{
                key: SimpleNamespace(
                    score=payload["score"],
                    probabilities=payload.get("probabilities", {}),
                    confidence=payload.get("confidence"),
                )
                for key, payload in (scores or {}).items()
            },
        }


class FakeTypesafeClient:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.calls = []

    def system_one(self, document, questions, *, model=None):
        self.calls.append(
            {
                "model": model,
                "document": document,
                "questions": questions,
            }
        )
        return self.evaluation


@typesafe_signature(
    fields={
        "severity": TypesafeFieldConfig(
            kind="score",
            score_levels={
                1: "Low urgency.",
                5: "Critical operational urgency.",
            },
        )
    }
)
class TicketPlanSignature(dspy.Signature):
    """Route an operational ticket."""

    ticket: str = dspy.InputField(desc="Ticket details.")
    customer_impact: bool = dspy.OutputField(desc="Whether the issue is customer impacting.")
    owner: Literal["api-platform", "checkout"] = dspy.OutputField(desc="The first-response owner team.")
    severity: float = dspy.OutputField(desc="Operational severity from 1 to 5.")
    summary: str = dspy.OutputField(desc="A short operational summary.")


def test_plan_signature_splits_typesafe_and_dspy_outputs():
    plan = plan_signature(TicketPlanSignature)

    assert plan.typesafe_field_names == ("customer_impact", "owner", "severity")
    assert plan.remaining_output_names == ("summary",)
    assert plan.reduced_signature is not None
    assert list(plan.reduced_signature.input_fields) == [
        "ticket",
        "customer_impact",
        "owner",
        "severity",
    ]
    assert list(plan.reduced_signature.output_fields) == ["summary"]


def test_typesafe_predict_merges_typesafe_outputs_with_dspy_fallback():
    class TicketRuntimeSignature(dspy.Signature):
        """Route an operational ticket."""

        title: str = dspy.InputField(desc="Ticket title.")
        customer_impact: bool = dspy.OutputField(desc="Whether the issue is customer impacting.")
        owner: Literal["api-platform", "checkout"] = dspy.OutputField(desc="The first-response owner team.")
        summary: str = dspy.OutputField(desc="A short operational summary.")

    client = FakeTypesafeClient(
        FakeEvaluation(
            nouls={"customer_impact": 0.91},
            choices={
                "owner": {
                    "choice": "option_1",
                    "probabilities": {
                        "option_0": 0.09,
                        "option_1": 0.91,
                    },
                    "confidence": 0.88,
                }
            },
        )
    )
    config = TypesafeConfig(
        client=client,
        model="speed_latest",
        prompt_factory=FakePromptFactory(),
    )

    dspy.configure(lm=DummyLM([{"summary": "Investigate the checkout latency spike."}]))
    predict = TypesafePredict(TicketRuntimeSignature, typesafe_config=config)

    result = predict(title="Intermittent API timeout in checkout")

    assert result.customer_impact is True
    assert result.owner == "checkout"
    assert result.summary == "Investigate the checkout latency spike."

    metadata = typesafe_results(result)
    assert metadata["customer_impact"].probability == 0.91
    assert metadata["owner"].probabilities == {
        "api-platform": 0.09,
        "checkout": 0.91,
    }
    timings = typesafe_timings(result)
    assert isinstance(timings, TypesafeTiming)
    assert timings.typesafe_seconds >= 0.0
    assert timings.dspy_seconds >= 0.0
    assert timings.total_seconds >= timings.typesafe_seconds

    assert client.calls[0]["model"] == "speed_latest"
    assert client.calls[0]["document"]["inputs"]["title"] == "Intermittent API timeout in checkout"
    assert list(client.calls[0]["questions"]) == ["customer_impact", "owner"]
    assert [question.kind for question in client.calls[0]["questions"].values()] == ["noul", "choice"]


def test_typesafe_predict_does_not_require_lm_for_fully_typesafe_signature():
    class RoutingSignature(dspy.Signature):
        """Choose the initial triage route."""

        ticket: str = dspy.InputField(desc="Ticket details.")
        customer_impact: bool = dspy.OutputField(desc="Whether the issue is customer impacting.")
        owner: Literal["api-platform", "checkout"] = dspy.OutputField(desc="The first-response owner team.")

    client = FakeTypesafeClient(
        FakeEvaluation(
            nouls={"customer_impact": 0.84},
            choices={
                "owner": {
                    "choice": "option_0",
                    "probabilities": {
                        "option_0": 0.84,
                        "option_1": 0.16,
                    },
                    "confidence": 0.84,
                }
            },
        )
    )
    config = TypesafeConfig(
        client=client,
        model="speed_latest",
        prompt_factory=FakePromptFactory(),
    )

    predict = TypesafePredict(RoutingSignature, typesafe_config=config)
    result = predict(ticket="Checkout latency spike during peak traffic")

    assert result.customer_impact is True
    assert result.owner == "api-platform"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_strict_predict_with_levels_needs_neither_lm_nor_patching(asynchronous):
    from dspy.signatures.field import DSPY_FIELD_ARG_NAMES

    class Review(dspy.Signature):
        text: str = dspy.InputField()
        supported: bool = dspy.OutputField()
        relevance: float = dspy.OutputField()

    methods = (dspy.Predict.forward, dspy.Predict.aforward, dspy.Predict._forward_preprocess)
    field_args = list(DSPY_FIELD_ARG_NAMES)
    client = FakeTypesafeClient(
        FakeEvaluation(
            nouls={"supported": 0.49},
            scores={"relevance": {"score": 1.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}},
        )
    )
    config = TypesafeConfig(client=client, model="jev-latest", prompt_factory=FakePromptFactory())
    predict = TypesafePredict(Review, config, strict=True, levels={"relevance": ["Unrelated", "Partial", "Direct"]})
    result = asyncio.run(predict.acall(text="Evidence")) if asynchronous else predict(text="Evidence")
    assert result.supported is False
    assert result.relevance == pytest.approx(1.6)
    assert typesafe_results(result)["relevance"].probabilities == {0: 0.1, 1: 0.2, 2: 0.7}
    assert len(client.calls) == 1
    assert client.calls[0]["questions"]["relevance"].levels == {0: "Unrelated", 1: "Partial", 2: "Direct"}
    assert config.field_configs == {}  # Constructor does not mutate shared configuration.
    assert methods == (dspy.Predict.forward, dspy.Predict.aforward, dspy.Predict._forward_preprocess)
    assert field_args == DSPY_FIELD_ARG_NAMES


def test_strict_predict_rejects_residual_outputs_and_call_overrides_before_inference():
    client = FakeTypesafeClient(FakeEvaluation())
    config = TypesafeConfig(client=client, model="jev-latest", prompt_factory=FakePromptFactory())
    for signature in ("text -> answer: str", "text -> supported: bool, score: float"):
        with pytest.raises(ValueError, match="Unsupported outputs"):
            TypesafePredict(signature, config, strict=True)
    predict = TypesafePredict("text -> supported: bool", config, strict=True)
    dspy.configure(lm=DummyLM([{"answer": "must not be used"}]))
    with pytest.raises(ValueError, match="Unsupported outputs"):
        predict(text="x", signature="text -> answer: str")
    with pytest.raises(ValueError, match="generation options"):
        predict(text="x", config={"temperature": 0.2})
    assert client.calls == []


@pytest.mark.parametrize(
    "levels",
    [
        [],
        {"unknown": ["Low", "High"]},
        {"text": ["Low", "High"]},
        {"score": []},
        {"score": ["Only"]},
        {"score": "Low, High"},
        {"score": ["Low", "Low"]},
        {"score": ["Low", " "]},
        {"score": ["Low", 2]},
    ],
)
def test_invalid_levels_rejected_before_inference(levels):
    client = FakeTypesafeClient(FakeEvaluation())
    config = TypesafeConfig(client=client, model="jev-latest", prompt_factory=FakePromptFactory())
    with pytest.raises(ValueError):
        TypesafePredict("text -> score: float", config, strict=True, levels=levels)
    assert client.calls == []


def test_levels_reject_conflicting_configuration():
    config = TypesafeConfig(
        client=FakeTypesafeClient(FakeEvaluation()),
        model="jev-latest",
        field_configs={"score": TypesafeFieldConfig(kind="score", score_levels={1: "Low", 5: "High"})},
    )
    with pytest.raises(ValueError, match="existing Typesafe override"):
        TypesafePredict("text -> score: float", config, levels={"score": ["Low", "High"]})


def test_enable_typesafe_wraps_existing_predictors():
    class RoutingSignature(dspy.Signature):
        """Choose the initial triage route."""

        ticket: str = dspy.InputField()
        customer_impact: bool = dspy.OutputField()
        owner: Literal["api-platform", "checkout"] = dspy.OutputField()

    class TicketProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.route = dspy.Predict(RoutingSignature)

    client = FakeTypesafeClient(FakeEvaluation())
    config = TypesafeConfig(
        client=client,
        model="speed_latest",
        prompt_factory=FakePromptFactory(),
    )

    program = TicketProgram()
    enable_typesafe(program, typesafe_config=config)

    assert isinstance(program.route, TypesafePredict)

    lm = DummyLM([{"owner": "checkout"}])
    program.set_lm(lm)
    assert program.route.lm is lm


def test_typesafeify_works_with_plain_dspy_predict():
    @typesafeify()
    class Emotion(dspy.Signature):
        """Classify emotion."""

        sentence: str = dspy.InputField()
        sentiment: Literal["sadness", "joy", "love", "anger", "fear", "surprise"] = dspy.OutputField()

    client = FakeTypesafeClient(
        FakeEvaluation(
            choices={
                "sentiment": {
                    "choice": "option_1",
                    "probabilities": {
                        "option_0": 0.03,
                        "option_1": 0.82,
                        "option_2": 0.05,
                        "option_3": 0.04,
                        "option_4": 0.03,
                        "option_5": 0.03,
                    },
                    "confidence": 0.82,
                }
            }
        )
    )

    configure_typesafe(
        client=client,
        model="speed_latest",
        prompt_factory=FakePromptFactory(),
    )

    predict = dspy.Predict(Emotion)
    result = predict(sentence="I got the job offer!")

    assert result.sentiment == "joy"
    assert typesafe_results(result)["sentiment"].confidence == 0.82
    assert client.calls[0]["document"]["inputs"]["sentence"] == "I got the job offer!"


def test_typesafeify_score_field_shorthand_builds_score_plan():
    @typesafeify(score_fields={"review_stars": [0, 5]})
    class ReviewSignature(dspy.Signature):
        """Estimate review quality."""

        review_text: str = dspy.InputField()
        review_stars: float = dspy.OutputField()

    plan = plan_signature(ReviewSignature)

    assert plan.typesafe_field_names == ("review_stars",)
    prompt_plan = plan.prompt_plans[0]
    assert prompt_plan.kind == "score"
    assert tuple(prompt_plan.score_levels) == (0, 2, 5)


def test_typesafe_score_maps_fuzzy_index_back_to_configured_scale():
    @typesafeify(score_fields={"severity": {1: "Low", 3: "Medium", 10: "Critical"}})
    class SeveritySignature(dspy.Signature):
        """Score operational severity."""

        ticket: str = dspy.InputField()
        severity: float = dspy.OutputField()

    client = FakeTypesafeClient(
        FakeEvaluation(
            scores={
                "severity": {
                    "score": 1.4,
                    "probabilities": {
                        0: 0.1,
                        1: 0.4,
                        2: 0.5,
                    },
                    "confidence": 0.8,
                }
            }
        )
    )
    configure_typesafe(
        client=client,
        model="speed_latest",
        prompt_factory=FakePromptFactory(),
    )

    result = dspy.Predict(SeveritySignature)(ticket="Checkout is unavailable")

    assert result.severity == 6.3
    assert typesafe_results(result)["severity"].expectation == 6.3
    assert typesafe_results(result)["severity"].probabilities == {
        1: 0.1,
        3: 0.4,
        10: 0.5,
    }


def test_typesafeify_can_clone_signature_for_side_by_side_comparison():
    class Emotion(dspy.Signature):
        """Classify emotion."""

        sentence: str = dspy.InputField()
        sentiment: Literal["sadness", "joy"] = dspy.OutputField()

    emotion_typesafe = typesafeify(
        Emotion,
        copy=True,
        signature_name="EmotionTypesafe",
    )

    assert emotion_typesafe is not Emotion
    assert emotion_typesafe.__name__ == "EmotionTypesafe"
    assert list(emotion_typesafe.fields) == list(Emotion.fields)
    assert not is_typesafeified(Emotion)
    assert is_typesafeified(emotion_typesafe)


def test_typesafeify_copy_preserves_existing_typesafe_metadata():
    @typesafeify()
    class Emotion(dspy.Signature):
        """Classify emotion."""

        sentence: str = dspy.InputField()
        sentiment: Literal["sadness", "joy"] = dspy.OutputField()

    emotion_clone = typesafeify(
        Emotion,
        copy=True,
        signature_name="EmotionClone",
    )

    plan = plan_signature(emotion_clone)

    assert is_typesafeified(emotion_clone)
    assert plan.typesafe_field_names == ("sentiment",)


def test_compare_predictions_renders_field_delta_table():
    class Emotion(dspy.Signature):
        """Classify emotion."""

        sentence: str = dspy.InputField()
        sentiment: Literal["sadness", "joy"] = dspy.OutputField()
        confidence: float = dspy.OutputField()

    baseline = dspy.Prediction(sentiment="sadness", confidence=0.24)
    candidate = dspy.Prediction(sentiment="joy", confidence=0.82)

    comparison = compare_predictions(Emotion, baseline, candidate)

    assert isinstance(comparison, PredictionComparison)
    assert comparison.changed_field_names == ("sentiment", "confidence")

    rendered = render_prediction_comparison(comparison, max_value_width=12)
    assert "field" in rendered
    assert "sentiment" in rendered
    assert "sadness" in rendered
    assert "joy" in rendered
    assert "+0.58" in rendered


def test_global_typesafe_config_does_not_intercept_undecorated_signatures_by_default():
    class Emotion(dspy.Signature):
        """Classify emotion."""

        sentence: str = dspy.InputField()
        sentiment: Literal["sadness", "joy"] = dspy.OutputField()

    client = FakeTypesafeClient(FakeEvaluation())
    configure_typesafe(
        client=client,
        model="speed_latest",
        prompt_factory=FakePromptFactory(),
    )

    dspy.configure(lm=DummyLM([{"sentiment": "sadness"}]))
    predict = dspy.Predict(Emotion)
    result = predict(sentence="This is bad.")

    assert result.sentiment == "sadness"
    assert client.calls == []

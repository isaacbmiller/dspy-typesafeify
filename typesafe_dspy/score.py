"""Rubric-bearing numeric output types for DSPy signatures."""

import math
from functools import lru_cache

from pydantic_core import core_schema


class Score(float):
    """A finite, zero-based rubric score: Score["Unrelated", "Partial", "Direct"].

    Arithmetic returns ordinary floats. The rubric belongs to the type, not to
    individual values; native probabilities remain in typesafe_results().
    """

    levels: tuple[str, ...] = ()

    def __new__(cls, value):
        if not cls.levels:
            raise TypeError("Use Score with at least two ordered descriptions")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Score requires a number, not a boolean or string")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= len(cls.levels) - 1:
            raise ValueError(f"Score must be finite and between 0 and {len(cls.levels) - 1}")
        return super().__new__(cls, value)

    def __class_getitem__(cls, levels):
        if cls is not Score:
            raise TypeError("Specialize Score, not an existing rubric")
        if (
            not isinstance(levels, tuple)
            or len(levels) < 2
            or any(not isinstance(level, str) or not level.strip() for level in levels)
            or len(set(levels)) != len(levels)
        ):
            raise ValueError("Score requires at least two distinct nonempty descriptions")
        return _score_type(levels)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        if not cls.levels:
            raise TypeError("Use Score with at least two ordered descriptions")
        return core_schema.no_info_before_validator_function(
            cls,
            core_schema.is_instance_schema(cls),
            json_schema_input_schema=core_schema.float_schema(ge=0, le=len(cls.levels) - 1, allow_inf_nan=False),
            serialization=core_schema.plain_serializer_function_ser_schema(float),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        return {
            "type": "number",
            "minimum": 0,
            "maximum": len(cls.levels) - 1,
            "description": "Ordered rubric: " + "; ".join(f"{i}: {level}" for i, level in enumerate(cls.levels)),
        }

    def __reduce__(self):
        return _restore_score, (self.levels, float(self))


@lru_cache(maxsize=128)
def _score_type(levels):
    return type("Score[" + ", ".join(repr(level) for level in levels) + "]", (Score,), {"levels": levels})


def _restore_score(levels, value):
    return Score[levels](value)

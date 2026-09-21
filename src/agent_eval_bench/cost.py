"""Token-cost accounting from a versioned list-price table.

Rule: tokens come from the agent's API-reported usage (via the adapter); dollars are computed
here from list prices. Unknown models yield ``None`` so the report can say "unknown" instead
of inventing a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import ModelCall, RunRecord, Verdict


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float
    effective: str | None = None
    provider: str = ""

    def cost(
        self, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int
    ) -> float:
        return (
            input_tokens * self.input
            + output_tokens * self.output
            + cache_read * self.cache_read
            + cache_write * self.cache_write
        ) / 1_000_000


class PriceTable:
    def __init__(self, prices: dict[str, Price], aliases: dict[str, str], version: str):
        self._prices = prices
        self._aliases = aliases
        self.version = version

    @classmethod
    def load(cls, path: Path) -> PriceTable:
        data = yaml.safe_load(Path(path).read_text()) or {}
        prices: dict[str, Price] = {}
        for provider, body in (data.get("providers") or {}).items():
            for model, row in (body.get("models") or {}).items():
                if not row or row.get("input") is None:
                    continue
                prices[model] = Price(
                    input=float(row["input"]),
                    output=float(row["output"]),
                    cache_read=float(row.get("cache_read", row["input"] * 0.1)),
                    cache_write=float(row.get("cache_write", row["input"] * 1.25)),
                    effective=row.get("effective"),
                    provider=provider,
                )
        return cls(prices, dict(data.get("aliases") or {}), str(data.get("version", "")))

    def resolve(self, model: str | None) -> str | None:
        if not model:
            return None
        if model in self._prices:
            return model
        if model in self._aliases and self._aliases[model] in self._prices:
            return self._aliases[model]
        # Tolerate dated suffixes like claude-sonnet-5-20260101 by longest-prefix match.
        best = None
        for known in self._prices:
            if model.startswith(known) and (best is None or len(known) > len(best)):
                best = known
        return best

    def price_for(self, model: str | None) -> Price | None:
        key = self.resolve(model)
        return self._prices.get(key) if key else None

    def known(self, model: str | None) -> bool:
        return self.price_for(model) is not None

    def call_cost(self, call: ModelCall, fallback_model: str | None = None) -> float | None:
        price = self.price_for(call.model or fallback_model)
        if price is None:
            return None
        return price.cost(
            call.input_tokens, call.output_tokens, call.cache_read_tokens, call.cache_write_tokens
        )

    def record_cost(self, record: RunRecord) -> tuple[float | None, bool]:
        """Return (cost, fully_known). Cost sums priced calls; unknown models make it partial."""
        total = 0.0
        known = True
        any_call = False
        for call in record.model_calls:
            any_call = True
            c = self.call_cost(call, record.model)
            if c is None:
                known = False
            else:
                total += c
        if not any_call:
            return (0.0, True)
        return (total, known)

    def record_cost_split(self, record: RunRecord) -> tuple[float, float, float] | None:
        """(input_cost, output_cost, total) for MLflow's cost attribute; None if any model is unknown."""
        inp = out = 0.0
        for call in record.model_calls:
            price = self.price_for(call.model or record.model)
            if price is None:
                return None
            inp += (
                call.input_tokens * price.input
                + call.cache_read_tokens * price.cache_read
                + call.cache_write_tokens * price.cache_write
            ) / 1_000_000
            out += call.output_tokens * price.output / 1_000_000
        return (inp, out, inp + out)

    def verdict_cost(self, v: Verdict) -> float | None:
        price = self.price_for(v.judge_model)
        if price is None:
            return None
        return price.cost(
            v.input_tokens, v.output_tokens, v.cache_read_tokens, v.cache_write_tokens
        )

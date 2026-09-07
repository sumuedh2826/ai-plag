from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

CREDITS_ENDPOINT = "/credits"
KEY_ENDPOINT = "/key"
HEADROOM_TIMEOUT_SECONDS = 20
# A run that fails partway costs money and leaves a half-written bank, so require
# headroom to exceed the projection by this factor before starting.
HEADROOM_SAFETY_FACTOR = 1.5


class HeadroomError(RuntimeError):
    pass


@dataclass(frozen=True)
class Headroom:
    account_credits: float
    account_usage: float
    key_limit: float | None
    key_usage: float
    key_usage_monthly: float

    @property
    def account_remaining(self) -> float:
        return self.account_credits - self.account_usage

    @property
    def key_remaining(self) -> float | None:
        if self.key_limit is None:
            return None
        return self.key_limit - self.key_usage

    @property
    def effective_remaining(self) -> float:
        """The binding constraint. An unlimited key falls back to account credit."""
        key_remaining = self.key_remaining
        if key_remaining is None:
            return self.account_remaining
        return min(key_remaining, self.account_remaining)

    def covers(self, projected_cost: float, safety_factor: float = HEADROOM_SAFETY_FACTOR) -> bool:
        return self.effective_remaining >= projected_cost * safety_factor


def fetch_headroom(api_key: str, base_url: str) -> Headroom:
    credits = _get_json(api_key, base_url, CREDITS_ENDPOINT).get("data") or {}
    key_info = _get_json(api_key, base_url, KEY_ENDPOINT).get("data") or {}
    raw_limit = key_info.get("limit")
    return Headroom(
        account_credits=float(credits.get("total_credits") or 0.0),
        account_usage=float(credits.get("total_usage") or 0.0),
        key_limit=None if raw_limit is None else float(raw_limit),
        key_usage=float(key_info.get("usage") or 0.0),
        key_usage_monthly=float(key_info.get("usage_monthly") or 0.0),
    )


def headroom_lines(headroom: Headroom, projected_cost: float) -> list[str]:
    key_limit = "unlimited" if headroom.key_limit is None else f"${headroom.key_limit:,.2f}"
    key_remaining = headroom.key_remaining
    key_remaining_text = "n/a" if key_remaining is None else f"${key_remaining:,.2f}"
    binding = "account credit" if headroom.key_remaining is None else (
        "key limit" if headroom.key_remaining <= headroom.account_remaining else "account credit"
    )
    return [
        "=== OpenRouter headroom ===",
        f"  account credits      ${headroom.account_credits:,.2f}",
        f"  account usage        ${headroom.account_usage:,.2f}",
        f"  account remaining    ${headroom.account_remaining:,.2f}",
        f"  key limit            {key_limit}",
        f"  key usage            ${headroom.key_usage:,.2f}",
        f"  key remaining        {key_remaining_text}",
        f"  key usage_monthly    ${headroom.key_usage_monthly:,.2f}",
        f"  binding constraint   {binding}",
        f"  effective remaining  ${headroom.effective_remaining:,.2f}",
        f"  projected run cost   ${projected_cost:,.2f}"
        f"  (x{HEADROOM_SAFETY_FACTOR} safety = ${projected_cost * HEADROOM_SAFETY_FACTOR:,.2f})",
    ]


def _get_json(api_key: str, base_url: str, path: str) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=HEADROOM_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HeadroomError(f"Could not read {path} from OpenRouter: {exc}") from exc
    if not isinstance(payload, dict):
        raise HeadroomError(f"Unexpected {path} payload from OpenRouter")
    return payload

"""Tests for Lichess rate-limit handling."""

from collections import defaultdict
from unittest.mock import Mock

from lib import lichess
from lib.timer import Timer, seconds


def make_lichess_without_init() -> lichess.Lichess:
    """Create a Lichess instance without validating a real token."""
    li = object.__new__(lichess.Lichess)
    li.rate_limit_timers = defaultdict(Timer)
    return li


def test_handle_challenge__generic_429_sets_challenge_rate_limit() -> None:
    """Test generic challenge 429 responses block the challenge endpoint briefly."""
    li = make_lichess_without_init()
    response = Mock(status_code=429)
    response.headers = {}
    response.json.return_value = {"error": "Too many requests. Try again later."}

    challenge_response = li.handle_challenge(response)

    assert challenge_response["bot_is_rate_limited"] is True
    assert challenge_response["rate_limit_timeout"] == seconds(60)
    assert li.is_rate_limited(lichess.ENDPOINTS["challenge"])


def test_handle_challenge__generic_429_uses_retry_after_header() -> None:
    """Test challenge 429 responses use Lichess' Retry-After header when present."""
    li = make_lichess_without_init()
    response = Mock(status_code=429)
    response.headers = {"Retry-After": "180"}
    response.json.return_value = {"error": "Too many requests. Try again later."}

    challenge_response = li.handle_challenge(response)

    assert challenge_response["bot_is_rate_limited"] is True
    assert challenge_response["rate_limit_timeout"] == seconds(180)
    assert li.rate_limit_time_left(lichess.ENDPOINTS["challenge"]) > seconds(170)

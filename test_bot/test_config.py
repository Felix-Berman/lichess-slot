"""Test functions for config module."""
import logging

import pytest

from lib import config


def test_config_assert__false() -> None:
    """Test that config_assert raises an exception with the provided error message."""
    with pytest.raises(Exception, match="some error"):
        config.config_assert(False, "some error")


def test_config_assert__true() -> None:
    """Test that config_assert does not raise when assertion is True."""
    config.config_assert(True, "no error")


def test_config_warn__true(caplog: pytest.LogCaptureFixture) -> None:
    """Test that config_warn does not log a warning when assertion is True."""
    with caplog.at_level(logging.WARNING):
        config.config_warn(True, "this should not appear")
        assert len(caplog.records) == 0  # No warning should be logged


def test_config_warn__false(caplog: pytest.LogCaptureFixture) -> None:
    """Test that config_warn logs a warning when assertion is False."""
    with caplog.at_level(logging.WARNING):
        config.config_warn(False, "test warning message")
        assert "test warning message" in caplog.text
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"


def test_insert_default_values__slot_matchmaking_does_not_inherit_global_time_controls() -> None:
    """Test slot matchmaking gets independent defaults and shared labelled pools."""
    config_dict = {
        "challenge": {},
        "correspondence": {},
        "matchmaking": {
            "allow_matchmaking": True,
            "challenge_days": [1, 2],
            "time_control_pools": {
                "short": [{"initial": 60, "increment": 0}],
            },
        },
        "slots": {
            "enabled": True,
            "definitions": {
                "short": {
                    "matchmaking": {
                        "challenge_time_control_pools": ["short"],
                    },
                },
                "human": {
                    "matchmaking": {
                        "allow_matchmaking": False,
                    },
                },
            },
        },
    }

    config.insert_default_values(config_dict)

    short_matchmaking = config_dict["slots"]["definitions"]["short"]["matchmaking"]
    human_matchmaking = config_dict["slots"]["definitions"]["human"]["matchmaking"]
    assert short_matchmaking["allow_matchmaking"] is True
    assert short_matchmaking["challenge_days"] == [None]
    assert short_matchmaking["time_control_pools"] == config_dict["matchmaking"]["time_control_pools"]
    assert human_matchmaking["allow_matchmaking"] is False

"""Test functions for matchmaking module."""
from unittest.mock import Mock
from lib.matchmaking import configured_real_time_controls, choose_time_control, game_category, Matchmaking
from lib.config import Configuration, insert_default_values
from lib.timer import seconds, minutes
from lib.lichess_types import UserProfileType


def test_game_category_standard_bullet() -> None:
    """Test bullet time control with config values."""
    # challenge_initial_time: 60 (1 min), challenge_increment: 1
    # 60 + 1*40 = 100 seconds < 179 = bullet
    assert game_category("standard", 60, 1, 0) == "bullet"

    # challenge_initial_time: 60, challenge_increment: 2
    # 60 + 2*40 = 140 seconds < 179 = bullet
    assert game_category("standard", 60, 2, 0) == "bullet"


def test_game_category_standard_blitz() -> None:
    """Test blitz time control with config values."""
    # challenge_initial_time: 180 (3 min), challenge_increment: 1
    # 180 + 1*40 = 220 seconds, 179 <= 220 < 479 = blitz
    assert game_category("standard", 180, 1, 0) == "blitz"

    # challenge_initial_time: 180, challenge_increment: 2
    # 180 + 2*40 = 260 seconds, 179 <= 260 < 479 = blitz
    assert game_category("standard", 180, 2, 0) == "blitz"


def test_game_category_standard_rapid() -> None:
    """Test rapid time control."""
    # 10 minutes + 5 seconds increment
    # 600 + 5*40 = 800 seconds, 479 <= 800 < 1499 = rapid
    assert game_category("standard", 600, 5, 0) == "rapid"

    # 15 minutes no increment
    # 900 + 0*40 = 900 seconds, 479 <= 900 < 1499 = rapid
    assert game_category("standard", 900, 0, 0) == "rapid"


def test_game_category_standard_classical() -> None:
    """Test classical time control with max config values."""
    # max_base: 1800 (30 min), max_increment: 20
    # 1800 + 20*40 = 2600 seconds >= 1499 = classical
    assert game_category("standard", 1800, 20, 0) == "classical"

    # 25 minutes no increment
    # 1500 + 0*40 = 1500 seconds >= 1499 = classical
    assert game_category("standard", 1500, 0, 0) == "classical"


def test_game_category_correspondence() -> None:
    """Test correspondence games with config values."""
    # min_days: 1
    assert game_category("standard", 0, 0, 1) == "correspondence"

    # challenge_days: 2
    assert game_category("standard", 0, 0, 2) == "correspondence"

    # max_days: 14
    assert game_category("standard", 0, 0, 14) == "correspondence"


def test_game_category_variants() -> None:
    """Test chess variants from config."""
    assert game_category("atomic", 60, 1, 0) == "atomic"
    assert game_category("chess960", 180, 2, 0) == "chess960"
    assert game_category("crazyhouse", 600, 5, 0) == "crazyhouse"
    assert game_category("horde", 60, 0, 0) == "horde"
    assert game_category("kingOfTheHill", 180, 1, 0) == "kingOfTheHill"
    assert game_category("racingKings", 600, 0, 0) == "racingKings"
    assert game_category("threeCheck", 60, 1, 0) == "threeCheck"
    assert game_category("antichess", 180, 2, 0) == "antichess"


def test_game_category_time_boundaries() -> None:
    """Test edge cases at time control boundaries."""
    # Exactly at bullet/blitz boundary
    # 179 seconds should be blitz (179 < 179 is False)
    assert game_category("standard", 179, 0, 0) == "blitz"

    # Just below boundary
    assert game_category("standard", 178, 0, 0) == "bullet"

    # Exactly at blitz/rapid boundary
    assert game_category("standard", 479, 0, 0) == "rapid"

    # Just below
    assert game_category("standard", 478, 0, 0) == "blitz"

    # Exactly at rapid/classical boundary
    assert game_category("standard", 1499, 0, 0) == "classical"

    # Just below
    assert game_category("standard", 1498, 0, 0) == "rapid"


def test_game_category_min_config_values() -> None:
    """Test minimum config values."""
    # min_base: 0, min_increment: 0
    # This is an edge case: 0 + 0*40 = 0 < 179 = bullet
    assert game_category("standard", 0, 0, 0) == "bullet"

    # min_base: 0, min_increment: 0, min_days: 1
    assert game_category("standard", 0, 0, 1) == "correspondence"


def test_game_category_correspondence_overrides_time() -> None:
    """Test that correspondence takes precedence over time controls."""
    # If both days and time controls are set, days takes precedence
    assert game_category("standard", 1800, 20, 1) == "correspondence"
    assert game_category("standard", 60, 1, 2) == "correspondence"


def test_game_category_variant_overrides_time() -> None:
    """Test that variants override time control categorization."""
    # Variants are returned regardless of time control
    # Even if time would be "classical", variant name is returned
    assert game_category("atomic", 1800, 20, 0) == "atomic"
    assert game_category("horde", 60, 1, 0) == "horde"

    # Variants override correspondence too
    assert game_category("chess960", 0, 0, 14) == "chess960"


def test_game_category_negative_values() -> None:
    """Test edge case with negative values (should not happen in practice)."""
    # Negative base time
    assert game_category("standard", -100, 5, 0) == "bullet"

    # Negative increment results in negative duration
    result = game_category("standard", 100, -10, 0)
    # 100 + (-10)*40 = -300, which is < 179, so bullet
    assert result == "bullet"


def test_game_category_realistic_scenarios() -> None:
    """Test realistic game scenarios from actual lichess games."""
    # 1+0 bullet
    assert game_category("standard", 60, 0, 0) == "bullet"

    # 2+1 bullet
    assert game_category("standard", 120, 1, 0) == "bullet"

    # 3+0 blitz
    assert game_category("standard", 180, 0, 0) == "blitz"

    # 3+2 blitz
    assert game_category("standard", 180, 2, 0) == "blitz"

    # 5+0 blitz
    assert game_category("standard", 300, 0, 0) == "blitz"

    # 5+3 blitz
    assert game_category("standard", 300, 3, 0) == "blitz"

    # 10+0 rapid
    assert game_category("standard", 600, 0, 0) == "rapid"

    # 15+5 rapid
    assert game_category("standard", 900, 5, 0) == "rapid"

    # 15+10 rapid
    assert game_category("standard", 900, 10, 0) == "rapid"

    # 30+0 classical
    assert game_category("standard", 1800, 0, 0) == "classical"

    # 30+20 classical
    assert game_category("standard", 1800, 20, 0) == "classical"


def test_get_random_config_value__returns_specific_value() -> None:
    """Test that get_random_config_value returns the config value when it's not 'random'."""
    # Create mock objects
    mock_li = Mock()
    mock_config = Configuration({
        "challenge": {"variants": ["standard"]},
        "matchmaking": {
            "allow_matchmaking": False,
            "block_list": [],
            "online_block_list": [],
            "challenge_timeout": 30
        }
    })
    mock_user_profile: UserProfileType = {"username": "testbot", "perfs": {}}

    # Create matchmaking instance
    matchmaking = Matchmaking(mock_li, mock_config, mock_user_profile)

    # Create config with a specific value
    test_config = Configuration({"challenge_variant": "atomic"})

    # Test that it returns the specific value, not a random choice
    choices = ["standard", "chess960", "atomic", "horde"]
    result = matchmaking.get_random_config_value(test_config, "challenge_variant", choices)

    assert result == "atomic", f"Expected 'atomic' but got '{result}'"


def test_get_random_config_value__returns_from_choices_when_random() -> None:
    """Test that get_random_config_value returns a value from choices when config value is 'random'."""
    # Create mock objects
    mock_li = Mock()
    mock_config = Configuration({
        "challenge": {"variants": ["standard"]},
        "matchmaking": {
            "allow_matchmaking": False,
            "block_list": [],
            "online_block_list": [],
            "challenge_timeout": 30
        }
    })
    mock_user_profile: UserProfileType = {"username": "testbot", "perfs": {}}

    # Create matchmaking instance
    matchmaking = Matchmaking(mock_li, mock_config, mock_user_profile)

    # Create config with "random" value
    test_config = Configuration({"challenge_mode": "random"})

    # Test that it returns one of the choices
    choices = ["casual", "rated"]
    result = matchmaking.get_random_config_value(test_config, "challenge_mode", choices)

    assert result in choices, f"Expected result to be in {choices} but got '{result}'"


def test_configured_real_time_controls__uses_explicit_pairs() -> None:
    """Test that paired time controls do not cross-combine initial and increment values."""
    match_config = Configuration({
        "challenge_initial_time": [30, 1800],
        "challenge_increment": [0, 30],
        "challenge_time_controls": [
            {"initial": 30, "increment": 0},
            {"initial": 1800, "increment": 30},
        ],
    })

    assert configured_real_time_controls(match_config) == [(30, 0), (1800, 30)]


def test_configured_real_time_controls__uses_selected_labelled_pools() -> None:
    """Test labelled pools let configs choose multiple independent groups."""
    match_config = Configuration({
        "challenge_initial_time": [30, 1800],
        "challenge_increment": [0, 30],
        "challenge_time_controls": [],
        "time_control_pools": {
            "short": [
                {"initial": 60, "increment": 0},
                {"initial": 120, "increment": 1},
            ],
            "long": [
                {"initial": 600, "increment": 5},
            ],
        },
        "challenge_time_control_pools": ["short", "long"],
    })

    assert configured_real_time_controls(match_config) == [(60, 0), (120, 1), (600, 5)]


def test_configured_real_time_controls__uses_pool_cross_product() -> None:
    """Test a labelled pool can define its own initial/increment product."""
    match_config = Configuration({
        "challenge_initial_time": [30],
        "challenge_increment": [30],
        "challenge_time_controls": [],
        "time_control_pools": {
            "increment": {
                "challenge_initial_time": [180, 300],
                "challenge_increment": [2, 3],
            },
        },
        "challenge_time_control_pools": ["increment"],
    })

    assert configured_real_time_controls(match_config) == [(180, 2), (180, 3), (300, 2), (300, 3)]


def test_configured_real_time_controls__falls_back_to_legacy_cross_product() -> None:
    """Test old initial/increment behavior remains available by default."""
    match_config = Configuration({
        "challenge_initial_time": [60, 180],
        "challenge_increment": [1, 2],
        "challenge_time_controls": [],
        "challenge_time_control_pools": [],
        "time_control_pools": {},
    })

    assert configured_real_time_controls(match_config) == [(60, 1), (60, 2), (180, 1), (180, 2)]


def test_choose_time_control__weights_correspondence_per_option(monkeypatch) -> None:
    """Test correspondence is one option among all configured controls, not a fixed 50% branch."""
    choices_seen = []

    def fake_choice(choices):  # noqa: ANN001
        choices_seen.append(choices)
        return choices[0]

    monkeypatch.setattr("lib.matchmaking.random.choice", fake_choice)
    match_config = Configuration({
        "challenge_initial_time": [60, 180],
        "challenge_increment": [0, 2],
        "challenge_time_controls": [],
        "challenge_time_control_pools": [],
        "time_control_pools": {},
        "challenge_days": [1],
    })

    assert choose_time_control(match_config) == (60, 0, 0)
    assert choices_seen[0] == ["clock", "clock", "clock", "clock", "correspondence"]


def correspondence_matchmaking_config(max_active_games: int = 3) -> Configuration:
    """Create a minimal config for correspondence matchmaking tests."""
    return Configuration({
        "challenge": {"variants": ["standard"]},
        "correspondence": {"max_active_games": max_active_games},
        "slots": {"enabled": True, "definitions": {}},
        "matchmaking": {
            "allow_matchmaking": True,
            "allow_during_games": True,
            "block_list": [],
            "online_block_list": [],
            "challenge_timeout": 1,
            "challenge_filter": "none",
            "challenge_variant": "standard",
            "challenge_mode": "rated",
            "challenge_initial_time": [60],
            "challenge_increment": [0],
            "challenge_time_controls": [],
            "challenge_time_control_pools": [],
            "challenge_days": [1],
            "opponent_min_rating": 600,
            "opponent_max_rating": 4000,
            "opponent_rating_difference": None,
            "rating_preference": "none",
            "overrides": {},
        },
    })


def test_challenge_correspondence__creates_daily_without_clock() -> None:
    """Test the correspondence scheduler creates daily challenges outside slots."""
    mock_li = Mock()
    mock_li.get_ongoing_games.return_value = []
    mock_li.get_online_bots.return_value = [
        {"username": "otherbot", "perfs": {"correspondence": {"games": 1, "rating": 1500}}},
    ]
    mock_li.get_public_data.return_value = {"blocking": False}
    mock_li.challenge.return_value = {"id": "abc123"}
    user_profile: UserProfileType = {"username": "testbot", "perfs": {"correspondence": {"rating": 1500}}}
    matchmaking = Matchmaking(mock_li, correspondence_matchmaking_config(), user_profile)
    matchmaking.min_wait_time = seconds(0)

    assert matchmaking.challenge_correspondence([]) is True
    mock_li.challenge.assert_called_once()
    _, params = mock_li.challenge.call_args.args
    assert params["days"] == 1
    assert "clock.limit" not in params
    assert "clock.increment" not in params


def test_challenge_correspondence__respects_correspondence_cap() -> None:
    """Test no daily challenge is created when the separate correspondence cap is full."""
    mock_li = Mock()
    mock_li.get_ongoing_games.return_value = [{"speed": "correspondence"}]
    user_profile: UserProfileType = {"username": "testbot", "perfs": {"correspondence": {"rating": 1500}}}
    matchmaking = Matchmaking(mock_li, correspondence_matchmaking_config(max_active_games=1), user_profile)
    matchmaking.min_wait_time = seconds(0)

    assert matchmaking.challenge_correspondence([]) is False
    mock_li.challenge.assert_not_called()


def test_challenge_slots__throttles_failed_opponent_search() -> None:
    """Test empty opponent searches consume the outgoing matchmaking cooldown."""
    mock_li = Mock()
    mock_li.get_online_bots.return_value = []
    user_profile: UserProfileType = {"username": "testbot", "perfs": {"bullet": {"rating": 1500}}}
    config_dict = {
        "challenge": {"variants": ["standard"]},
        "correspondence": {"max_active_games": 3},
        "matchmaking": {
            "allow_matchmaking": True,
            "allow_during_games": True,
            "challenge_timeout": 1,
            "challenge_variant": "standard",
            "challenge_mode": "rated",
            "challenge_initial_time": [60],
            "challenge_increment": [0],
            "challenge_days": [],
            "opponent_rating_difference": None,
        },
        "slots": {
            "enabled": True,
            "definitions": {
                "short": {
                    "matchmaking": {
                        "challenge_initial_time": [60],
                        "challenge_increment": [0],
                        "challenge_days": [],
                    },
                },
                "long": {
                    "matchmaking": {
                        "challenge_initial_time": [60],
                        "challenge_increment": [0],
                        "challenge_days": [],
                    },
                },
            },
        },
    }
    insert_default_values(config_dict)
    matchmaking = Matchmaking(mock_li, Configuration(config_dict), user_profile)
    matchmaking.last_challenge_created_delay.starting_time -= minutes(2).total_seconds()

    assert matchmaking.challenge_slots(set(), [], 3, {}) is False
    mock_li.get_online_bots.assert_called_once()
    mock_li.challenge.assert_not_called()
    assert not matchmaking.should_create_slot_challenge()

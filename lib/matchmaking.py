"""Challenge other bots."""
import random
import logging
import datetime
import contextlib
from lib import model
from lib.timer import Timer, days, seconds, minutes, years
from collections import defaultdict
from collections.abc import Sequence
from lib.lichess import ENDPOINTS, Lichess, RateLimitedError
from lib.config import Configuration
from typing import Any, cast, TypeAlias
from lib.blocklist import OnlineBlocklist
from lib.lichess_types import UserProfileType, PerfType, EventType, FilterType, ChallengeType, GameType
MULTIPROCESSING_LIST_TYPE: TypeAlias = Sequence[model.Challenge]

logger = logging.getLogger(__name__)


class Matchmaking:
    """Challenge other bots."""

    def __init__(self, li: Lichess, config: Configuration, user_profile: UserProfileType) -> None:
        """Initialize values needed for matchmaking."""
        self.li = li
        self.variants = list(filter(lambda variant: variant != "fromPosition", config.challenge.variants))
        self.matchmaking_cfg = config.matchmaking
        self.correspondence_cfg = config.correspondence
        self.slots_cfg = config.lookup("slots") or Configuration({"enabled": False, "definitions": {}})
        self.user_profile = user_profile
        self.last_challenge_created_delay = Timer(seconds(25))  # Challenges expire after 20 seconds.
        self.last_game_ended_delay = Timer(minutes(self.matchmaking_cfg.challenge_timeout))
        self.last_user_profile_update_time = Timer(minutes(5))
        self.min_wait_time = minutes(self.matchmaking_cfg.challenge_timeout)
        self.rate_limit_timer = Timer()
        self.slot_game_done_delays: dict[str, Timer] = {}
        self.slot_challenge_ids: dict[str, str] = {}
        self.correspondence_challenge_id = ""

        # Maximum time between challenges, even if there are active games
        self.max_wait_time = minutes(10) if self.matchmaking_cfg.allow_during_games else years(10)
        self.challenge_id = ""

        # (opponent name, game aspect) --> other bot is likely to accept challenge
        # game aspect is the one the challenged bot objects to and is one of:
        #   - game speed (bullet, blitz, etc.)
        #   - variant (standard, horde, etc.)
        #   - casual/rated
        #   - empty string (if no other reason is given or self.filter_type is COARSE)
        self.challenge_type_acceptable: defaultdict[tuple[str, str], Timer] = defaultdict(Timer)
        self.challenge_filter = self.matchmaking_cfg.challenge_filter

        for name in self.matchmaking_cfg.block_list:
            self.add_to_block_list(name)

        self.online_block_list = OnlineBlocklist(self.matchmaking_cfg.online_block_list)

    def should_create_challenge(self) -> bool:
        """Whether we should create a challenge."""
        matchmaking_enabled = self.matchmaking_cfg.allow_matchmaking
        time_has_passed = self.last_game_ended_delay.is_expired() and self.rate_limit_timer.is_expired()
        challenge_expired = self.last_challenge_created_delay.is_expired() and self.challenge_id
        min_wait_time_passed = self.last_challenge_created_delay.time_since_reset() > self.min_wait_time
        if challenge_expired:
            self.li.cancel(self.challenge_id)
            logger.info(f"Challenge id {self.challenge_id} cancelled.")
            self.discard_challenge(self.challenge_id)
            self.show_earliest_challenge_time()
        return bool(matchmaking_enabled and (time_has_passed or challenge_expired) and min_wait_time_passed)

    def should_create_slot_challenge(self, game_count: int = 0) -> bool:
        """Whether a matchmaking slot may create a challenge."""
        matchmaking_enabled = self.matchmaking_cfg.allow_matchmaking
        min_wait_time_passed = self.last_challenge_created_delay.time_since_reset() > self.min_wait_time
        active_game_wait_passed = (game_count == 0
                                   or self.last_challenge_created_delay.time_since_reset() >= self.max_wait_time)
        return bool(matchmaking_enabled
                    and self.rate_limit_timer.is_expired()
                    and active_game_wait_passed
                    and min_wait_time_passed)

    def create_challenge(self, username: str, base_time: int, increment: int, days: int, variant: str,
                         mode: str) -> str:
        """Create a challenge."""
        params: dict[str, str | int | bool] = {"rated": mode == "rated", "variant": variant}

        if days:
            params["days"] = days
        elif base_time or increment:
            params["clock.limit"] = base_time
            params["clock.increment"] = increment
        else:
            logger.error("At least one of challenge_days, challenge_initial_time, or challenge_increment "
                         "must be greater than zero in the matchmaking section of your config file.")
            return ""

        try:
            self.last_challenge_created_delay.reset()
            response = self.li.challenge(username, params)
            challenge_id = response.get("id", "")
            if not challenge_id:
                self.handle_challenge_error_response(response, username)
            return challenge_id
        except RateLimitedError as e:
            self.report_bot_rate_limit(str(e), e.timeout)
        except Exception as e:
            logger.debug(e, exc_info=e)

        logger.warning("Could not create challenge")
        self.show_earliest_challenge_time()
        return ""

    def handle_challenge_error_response(self, response: ChallengeType, username: str) -> None:
        """If a challenge fails, print the error and adjust the challenge requirements in response."""
        logger.error(response)
        if response.get("bot_is_rate_limited"):
            timeout = cast(datetime.timedelta, response.get("rate_limit_timeout"))
            self.rate_limit_timer = Timer(timeout)
            if not response.get("ratelimit"):
                self.add_challenge_filter(username, "")
        elif response.get("opponent_is_rate_limited"):
            self.add_challenge_filter(username, "", response.get("rate_limit_timeout"))
        else:
            self.add_challenge_filter(username, "")
        self.show_earliest_challenge_time()

    def report_bot_rate_limit(self, error: str, timeout: datetime.timedelta) -> None:
        """Report that our bot is rate limited and pause outgoing matchmaking."""
        response: ChallengeType = {
            "error": error,
            "bot_is_rate_limited": True,
            "opponent_is_rate_limited": False,
            "rate_limit_timeout": timeout,
        }
        logger.error(response)
        self.rate_limit_timer = Timer(timeout)
        self.show_earliest_challenge_time()

    def report_endpoint_rate_limit(self, endpoint_name: str) -> bool:
        """Report an existing Lichess endpoint cooldown if it blocks matchmaking."""
        path_template = ENDPOINTS[endpoint_name]
        if self.li.is_rate_limited(path_template) is not True:
            return False

        timeout = self.li.rate_limit_time_left(path_template)
        self.report_bot_rate_limit(f"Endpoint {path_template} is rate limited. Try again later.", timeout)
        return True

    def record_matchmaking_attempt(self) -> None:
        """Throttle outgoing matchmaking attempts, even when no opponent is available."""
        self.last_challenge_created_delay.reset()

    def perf(self) -> dict[str, PerfType]:
        """Get the bot's rating in every variant. Bullet, blitz, rapid etc. are considered different variants."""
        user_perf: dict[str, PerfType] = self.user_profile["perfs"]
        return user_perf

    def username(self) -> str:
        """Our username."""
        username: str = self.user_profile["username"]
        return username

    def update_user_profile(self) -> None:
        """Update our user profile data, to get our latest rating."""
        if self.last_user_profile_update_time.is_expired():
            self.last_user_profile_update_time.reset()
            with contextlib.suppress(Exception):
                self.user_profile = self.li.get_profile()

    def get_weights(self, online_bots: list[UserProfileType], rating_preference: str, min_rating: int, max_rating: int,
                    game_type: str) -> list[int]:
        """Get the weight for each bot. A higher weights means the bot is more likely to get challenged."""
        def rating(bot: UserProfileType) -> int:
            perfs: dict[str, PerfType] = bot.get("perfs", {})
            perf: PerfType = perfs.get(game_type, {})
            return perf.get("rating", 0)

        if rating_preference == "high":
            # A bot with max_rating rating will be twice as likely to get picked than a bot with min_rating rating.
            reduce_ratings_by = min(min_rating - (max_rating - min_rating), min_rating - 1)
            weights = [rating(bot) - reduce_ratings_by for bot in online_bots]
        elif rating_preference == "low":
            # A bot with min_rating rating will be twice as likely to get picked than a bot with max_rating rating.
            reduce_ratings_by = max(max_rating - (min_rating - max_rating), max_rating + 1)
            weights = [reduce_ratings_by - rating(bot) for bot in online_bots]
        else:
            weights = [1] * len(online_bots)
        return weights

    def choose_matchmaking_config(self) -> tuple[str, Configuration]:
        """Choose the matchmaking configuration to use for one outgoing challenge."""
        override_choice = random.choice(self.matchmaking_cfg.overrides.keys() + [None])
        logger.info(f"Using the {override_choice or 'default'} matchmaking configuration.")
        override = {} if override_choice is None else self.matchmaking_cfg.overrides.lookup(override_choice)
        return override_choice or "default", self.matchmaking_cfg | override

    def choose_opponent(self, match_config: Configuration | None = None) -> tuple[str | None, int, int, int, str, str]:
        """Choose an opponent."""
        if match_config is None:
            _, match_config = self.choose_matchmaking_config()

        variant = self.get_random_config_value(match_config, "challenge_variant", self.variants)
        mode = self.get_random_config_value(match_config, "challenge_mode", ["casual", "rated"])
        rating_preference = match_config.rating_preference

        base_time, increment, num_days = choose_time_control(match_config)

        game_type = game_category(variant, base_time, increment, num_days)

        min_rating = match_config.opponent_min_rating
        max_rating = match_config.opponent_max_rating
        rating_diff = match_config.opponent_rating_difference
        bot_rating = self.perf().get(game_type, {}).get("rating", 0)
        if rating_diff is not None and bot_rating > 0:
            min_rating = bot_rating - rating_diff
            max_rating = bot_rating + rating_diff
        logger.info(f"Seeking {game_type} game with opponent rating in [{min_rating}, {max_rating}] ...")

        def is_suitable_opponent(bot: UserProfileType) -> bool:
            perf = bot.get("perfs", {}).get(game_type, {})
            return (bot["username"] != self.username()
                    and not self.in_block_list(bot["username"])
                    and perf.get("games", 0) > 0
                    and min_rating <= perf.get("rating", 0) <= max_rating)

        self.online_block_list.refresh()
        if self.report_endpoint_rate_limit("online_bots"):
            return None, base_time, increment, num_days, variant, mode
        online_bots = self.li.get_online_bots()
        if self.report_endpoint_rate_limit("online_bots"):
            return None, base_time, increment, num_days, variant, mode
        online_bots = list(filter(is_suitable_opponent, online_bots))

        def ready_for_challenge(bot: UserProfileType) -> bool:
            aspects = [variant, game_type, mode] if self.challenge_filter == FilterType.FINE else []
            return all(self.should_accept_challenge(bot["username"], aspect) for aspect in aspects)

        ready_bots = list(filter(ready_for_challenge, online_bots))
        online_bots = ready_bots or online_bots
        bot_username = None
        weights = self.get_weights(online_bots, rating_preference, min_rating, max_rating, game_type)

        try:
            bot = random.choices(online_bots, weights=weights)[0]
            bot_profile = self.li.get_public_data(bot["username"])
            if bot_profile.get("blocking"):
                self.add_to_block_list(bot["username"])
            else:
                bot_username = bot["username"]
        except Exception:
            if self.report_endpoint_rate_limit("public_data"):
                pass
            elif online_bots:
                logger.exception("Error:")
            else:
                logger.error("No suitable bots found to challenge.")

        return bot_username, base_time, increment, num_days, variant, mode

    def get_random_config_value(self, config: Configuration, parameter: str, choices: list[str]) -> str:
        """Choose a random value from `choices` if the parameter value in the config is `random`."""
        value: str = config.lookup(parameter)
        return value if value != "random" else random.choice(choices)

    def challenge(self,
                  active_games: set[str],
                  challenge_queue: MULTIPROCESSING_LIST_TYPE,
                  max_games: int,
                  slot_assignments: dict[str, str] | None = None) -> None:
        """
        Challenge an opponent.

        :param active_games: The games that the bot is playing.
        :param challenge_queue: The queue containing the challenges.
        :param max_games: The maximum allowed number of simultaneous games.
        """
        if self.slots_cfg.enabled:
            challenge_created = self.challenge_slots(active_games, challenge_queue, max_games, slot_assignments or {})
            if not challenge_created:
                self.challenge_correspondence(challenge_queue, active_games)
            return

        max_games_for_matchmaking = max_games if self.matchmaking_cfg.allow_during_games else min(1, max_games)
        game_count = len(active_games) + len(challenge_queue)
        if (game_count >= max_games_for_matchmaking
                or (game_count > 0 and self.last_challenge_created_delay.time_since_reset() < self.max_wait_time)
                or not self.should_create_challenge()):
            return

        logger.info("Challenging a random bot")
        self.record_matchmaking_attempt()
        self.update_user_profile()
        bot_username, base_time, increment, days, variant, mode = self.choose_opponent()
        if not bot_username:
            return

        logger.info(f"Will challenge {bot_username} for a {variant} game.")
        challenge_id = self.create_challenge(bot_username, base_time, increment, days, variant, mode)
        logger.info(f"Challenge id is {challenge_id or 'None'}.")
        self.challenge_id = challenge_id

    def challenge_slots(self,
                        active_games: set[str],
                        challenge_queue: MULTIPROCESSING_LIST_TYPE,
                        max_games: int,
                        slot_assignments: dict[str, str]) -> bool:
        """Challenge opponents using independently throttled matchmaking slots."""
        slots = self.slots_cfg.definitions
        if not slots:
            logger.warning("slots.enabled is true, but no slots are configured.")
            return False

        global_capacity = max_games if self.matchmaking_cfg.allow_during_games else min(1, max_games)
        if len(active_games) + len(challenge_queue) >= global_capacity:
            return False

        for slot_name in slots.keys():
            if self.challenge_slot(slot_name,
                                   slots.lookup(slot_name),
                                   active_games,
                                   challenge_queue,
                                   global_capacity,
                                   slot_assignments):
                return True
        return False

    def challenge_slot(self, slot_name: str, slot_config: Configuration, active_games: set[str],
                       challenge_queue: MULTIPROCESSING_LIST_TYPE, global_capacity: int,
                       slot_assignments: dict[str, str]) -> bool:
        """Try to create one challenge for a slot."""
        challenge_id = self.slot_challenge_ids.get(slot_name, "")
        if challenge_id and self.last_challenge_created_delay.is_expired():
            self.li.cancel(challenge_id)
            logger.info(f"Challenge id {challenge_id} cancelled.")
            self.slot_challenge_ids[slot_name] = ""
            challenge_id = ""

        if challenge_id:
            return False

        if not slot_config.matchmaking.allow_matchmaking:
            return False

        slot_delay = self.slot_game_done_delays.get(slot_name)
        occupied_slots = len(active_games) + len(challenge_queue) + sum(bool(challenge_id)
                                                                       for challenge_id in self.slot_challenge_ids.values())
        slot_occupied = (sum(assigned_slot == slot_name for assigned_slot in slot_assignments.values())
                         + sum(bool(challenge_id) for challenge_id_slot, challenge_id in self.slot_challenge_ids.items()
                               if challenge_id_slot == slot_name))
        if (occupied_slots >= global_capacity
                or slot_occupied >= slot_config.concurrency
                or (slot_delay is not None and not slot_delay.is_expired())
                or not self.should_create_slot_challenge(len(active_games) + len(challenge_queue))):
            return False

        logger.info(f"Challenging a random bot for slot {slot_name}")
        self.record_matchmaking_attempt()
        self.update_user_profile()
        match_config = slot_config.matchmaking
        bot_username, base_time, increment, days, variant, mode = self.choose_opponent(match_config)
        if not bot_username:
            return False

        logger.info(f"Will challenge {bot_username} for a {variant} game in slot {slot_name}.")
        new_challenge_id = self.create_challenge(bot_username, base_time, increment, days, variant, mode)
        logger.info(f"Challenge id is {new_challenge_id or 'None'}.")
        self.slot_challenge_ids[slot_name] = new_challenge_id
        return bool(new_challenge_id)

    def challenge_correspondence(self,
                                 challenge_queue: MULTIPROCESSING_LIST_TYPE,
                                 active_games: set[str] | None = None) -> bool:
        """Create a correspondence challenge without occupying a compute slot."""
        if challenge_queue:
            return False

        challenge_id = self.correspondence_challenge_id
        if challenge_id and self.last_challenge_created_delay.is_expired():
            self.li.cancel(challenge_id)
            logger.info(f"Challenge id {challenge_id} cancelled.")
            self.discard_challenge(challenge_id)

        if self.correspondence_challenge_id:
            return False

        if not configured_correspondence_days(self.matchmaking_cfg):
            return False

        active_game_count = len(active_games) if active_games is not None else 0
        game_count = active_game_count + len(challenge_queue)
        if (self.correspondence_game_count(challenge_queue) >= self.correspondence_cfg.max_active_games
                or not self.should_create_slot_challenge(game_count)):
            return False

        logger.info("Challenging a random bot for a correspondence game")
        self.record_matchmaking_attempt()
        self.update_user_profile()
        match_config = self.correspondence_matchmaking_config()
        bot_username, base_time, increment, days, variant, mode = self.choose_opponent(match_config)
        if not bot_username:
            return False

        logger.info(f"Will challenge {bot_username} for a {variant} correspondence game.")
        new_challenge_id = self.create_challenge(bot_username, base_time, increment, days, variant, mode)
        logger.info(f"Challenge id is {new_challenge_id or 'None'}.")
        self.correspondence_challenge_id = new_challenge_id
        return bool(new_challenge_id)

    def correspondence_matchmaking_config(self) -> Configuration:
        """Return a matchmaking config constrained to correspondence time controls."""
        config = self.matchmaking_cfg.config | {
            "challenge_initial_time": [None],
            "challenge_increment": [None],
            "challenge_time_controls": [],
            "challenge_time_control_pools": [],
        }
        return Configuration(config)

    def correspondence_game_count(self, challenge_queue: MULTIPROCESSING_LIST_TYPE) -> int:
        """Count current and queued correspondence games/challenges."""
        active_games: list[GameType] = self.li.get_ongoing_games() or []
        active_correspondence_games = sum(game.get("speed") == "correspondence" for game in active_games)
        queued_correspondence_challenges = sum(challenge.speed == "correspondence" for challenge in challenge_queue)
        pending_correspondence_challenge = 1 if self.correspondence_challenge_id else 0
        return active_correspondence_games + queued_correspondence_challenges + pending_correspondence_challenge

    def discard_challenge(self, challenge_id: str) -> None:
        """
        Clear the ID of the most recent challenge if it is no longer needed.

        :param challenge_id: The ID of the challenge that is expired, accepted, or declined.
        """
        if self.challenge_id == challenge_id:
            self.challenge_id = ""
        if self.correspondence_challenge_id == challenge_id:
            self.correspondence_challenge_id = ""
        for slot_name, slot_challenge_id in list(self.slot_challenge_ids.items()):
            if slot_challenge_id == challenge_id:
                self.slot_challenge_ids[slot_name] = ""

    def slot_for_challenge(self, challenge_id: str) -> str | None:
        """Return the slot that created a pending outgoing challenge."""
        for slot_name, slot_challenge_id in self.slot_challenge_ids.items():
            if slot_challenge_id == challenge_id:
                return slot_name
        return None

    def game_done(self) -> None:
        """Reset the timer for when the last game ended, and prints the earliest that the next challenge will be created."""
        self.last_game_ended_delay.reset()
        self.show_earliest_challenge_time()

    def slot_game_done(self, slot_name: str | None) -> None:
        """Reset the per-slot timer when a slot finishes using compute."""
        if slot_name is not None:
            slot_config = self.slots_cfg.definitions.lookup(slot_name)
            slot_timeout = slot_config.challenge_timeout if slot_config else self.matchmaking_cfg.challenge_timeout
            self.slot_game_done_delays[slot_name] = Timer(minutes(slot_timeout))

    def show_earliest_challenge_time(self) -> None:
        """Show the earliest that the next challenge will be created."""
        if self.matchmaking_cfg.allow_matchmaking:
            postgame_timeout = self.last_game_ended_delay.time_until_expiration()
            time_to_next_challenge = self.min_wait_time - self.last_challenge_created_delay.time_since_reset()
            rate_limit_delay = self.rate_limit_timer.time_until_expiration()
            time_left = max(postgame_timeout, time_to_next_challenge, rate_limit_delay)
            earliest_challenge_time = datetime.datetime.now() + time_left
            logger.info(f"Next challenge will be created after {earliest_challenge_time.strftime('%c')}")

    def add_to_block_list(self, username: str) -> None:
        """Add a bot to the blocklist."""
        self.add_challenge_filter(username, "", years(10))

    def in_block_list(self, username: str) -> bool:
        """Check if an opponent is in the block list to prevent future challenges."""
        return (not self.should_accept_challenge(username, "")) or username in self.online_block_list

    def add_challenge_filter(self, username: str, game_aspect: str, timeout: datetime.timedelta | None = None) -> None:
        """
        Prevent creating another challenge for a timeout when an opponent has declined a challenge.

        :param username: The name of the opponent.
        :param game_aspect: The aspect of a game (time control, chess variant, etc.) that caused the opponent to decline a
        challenge. If the parameter is empty, that is equivalent to adding the opponent to the block list.
        :param timeout: The amount of time to not challenge an opponent. If None, the default is a day.
        """
        self.challenge_type_acceptable[(username, game_aspect)] = Timer(timeout or days(1))

    def should_accept_challenge(self, username: str, game_aspect: str) -> bool:
        """
        Whether a bot is likely to accept a challenge to a game.

        :param username: The name of the opponent.
        :param game_aspect: A category of the challenge type (time control, chess variant, etc.) to test for acceptance.
        If game_aspect is empty, this is equivalent to checking if the opponent is in the block list.
        """
        return self.challenge_type_acceptable[(username, game_aspect)].is_expired()

    def accepted_challenge(self, event: EventType) -> None:
        """
        Set the challenge id to an empty string, if the challenge was accepted.

        Otherwise, we would attempt to cancel the challenge later.
        """
        self.discard_challenge(event["game"]["id"])

    def declined_challenge(self, event: EventType) -> None:
        """
        Handle a challenge that was declined by the opponent.

        Depends on whether `FilterType` is `NONE`, `COARSE`, or `FINE`.
        """
        challenge = model.Challenge(event["challenge"], self.user_profile)
        opponent = challenge.challenge_target
        reason = event["challenge"]["declineReason"]
        logger.info(f"{opponent} declined {challenge}: {reason}")
        self.discard_challenge(challenge.id)
        if not challenge.from_self or self.challenge_filter == FilterType.NONE:
            return

        mode = "rated" if challenge.rated else "casual"
        decline_details: dict[str, str] = {"generic": "",
                                           "later": "",
                                           "nobot": "",
                                           "toofast": challenge.speed,
                                           "tooslow": challenge.speed,
                                           "timecontrol": challenge.speed,
                                           "rated": mode,
                                           "casual": mode,
                                           "standard": challenge.variant,
                                           "variant": challenge.variant}

        reason_key = event["challenge"]["declineReasonKey"].lower()
        if reason_key not in decline_details:
            logger.warning(f"Unknown decline reason received: {reason_key}")
        game_problem = decline_details.get(reason_key, "") if self.challenge_filter == FilterType.FINE else ""
        self.add_challenge_filter(opponent.name, game_problem)
        logger.info(f"Will not challenge {opponent} to another {game_problem}".strip() + " game today.")

        self.show_earliest_challenge_time()


def game_category(variant: str, base_time: int, increment: int, num_days: int) -> str:
    """
    Get the game type (e.g. bullet, atomic, classical). Lichess has one rating for every variant regardless of time control.

    :param variant: The game's variant.
    :param base_time: The base time in seconds.
    :param increment: The increment in seconds.
    :param num_days: If the game is correspondence, we have some days to play the move.
    :return: The game category.
    """
    game_duration = base_time + increment * 40
    if variant != "standard":
        return variant
    if num_days:
        return "correspondence"
    if game_duration < 179:
        return "bullet"
    if game_duration < 479:
        return "blitz"
    if game_duration < 1499:
        return "rapid"
    return "classical"


def choose_time_control(match_config: Configuration) -> tuple[int, int, int]:
    """Choose a time control from labelled pools, paired controls, or legacy matchmaking config."""
    real_time_controls = configured_real_time_controls(match_config)
    correspondence_days = configured_correspondence_days(match_config)

    choices: list[str] = []
    choices.extend(["clock"] * len(real_time_controls))
    choices.extend(["correspondence"] * len(correspondence_days))
    if not choices:
        return 0, 0, 0

    if random.choice(choices) == "correspondence":
        return 0, 0, random.choice(correspondence_days)

    base_time, increment = random.choice(real_time_controls)
    return base_time, increment, 0


def configured_real_time_controls(match_config: Configuration) -> list[tuple[int, int]]:
    """Return valid real-time controls as explicit (initial, increment) pairs."""
    time_controls: list[tuple[int, int]] = []
    time_controls.extend(configured_pool_time_controls(match_config))

    for entry in match_config.lookup("challenge_time_controls") or []:
        time_control = normalize_time_control(entry)
        if time_control:
            time_controls.append(time_control)

    if time_controls:
        return time_controls

    initial_times = [initial_time for initial_time in match_config.challenge_initial_time or [] if initial_time is not None]
    increments = [increment for increment in match_config.challenge_increment or [] if increment is not None]
    return [(initial_time, increment) for initial_time in initial_times for increment in increments]


def configured_pool_time_controls(match_config: Configuration) -> list[tuple[int, int]]:
    """Return explicit controls from selected labelled time-control pools."""
    time_control_pools = match_config.lookup("time_control_pools")
    if not isinstance(time_control_pools, Configuration):
        return []

    time_controls: list[tuple[int, int]] = []
    for pool_name in match_config.lookup("challenge_time_control_pools") or []:
        pool = time_control_pools.lookup(pool_name)
        time_controls.extend(normalize_time_control_pool(pool))
    return time_controls


def normalize_time_control_pool(pool: Any) -> list[tuple[int, int]]:
    """Normalize one labelled time-control pool."""
    if isinstance(pool, Configuration):
        pool = pool.config

    if isinstance(pool, dict):
        explicit_controls = pool.get("challenge_time_controls", pool.get("time_controls"))
        if explicit_controls:
            return list(filter(None, (normalize_time_control(entry) for entry in explicit_controls)))

        initial_times = pool.get("challenge_initial_time", pool.get("initial_time", pool.get("initial", []))) or []
        increments = pool.get("challenge_increment", pool.get("increment", [])) or []
        if not isinstance(initial_times, list):
            initial_times = [initial_times]
        if not isinstance(increments, list):
            increments = [increments]
        return [(int(initial_time), int(increment))
                for initial_time in initial_times
                for increment in increments
                if initial_time is not None and increment is not None]

    if isinstance(pool, list):
        return list(filter(None, (normalize_time_control(entry) for entry in pool)))

    return []


def configured_correspondence_days(match_config: Configuration) -> list[int]:
    """Return configured correspondence days."""
    return [days for days in match_config.challenge_days or [] if days is not None]


def normalize_time_control(entry: Any) -> tuple[int, int] | None:
    """Normalize a time control configured as a dict or a two-item list."""
    if isinstance(entry, dict):
        initial = entry.get("initial", entry.get("base"))
        increment = entry.get("increment")
    elif isinstance(entry, list | tuple) and len(entry) == 2:
        initial, increment = entry
    else:
        return None

    if initial is None or increment is None:
        return None

    return int(initial), int(increment)

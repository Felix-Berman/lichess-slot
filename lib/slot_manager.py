"""Per-slot matchmaking manager for lichess-bot.

Each concurrency slot can have its own independent matchmaking configuration,
challenge acceptance constraints, and challenge timer. Slots operate autonomously:
no slot blocks another from sending outgoing challenges.

This module is specific to this fork and does not exist in upstream lichess-bot.
When rebasing, this file can be kept as-is since it introduces no changes to
existing files' internal logic.
"""
from __future__ import annotations
import logging
import multiprocessing
import multiprocessing.synchronize
from dataclasses import dataclass, field
from lib import matchmaking as matchmaking_module
from lib.config import Configuration
from lib.lichess import Lichess
from lib.lichess_types import UserProfileType, EventType
from lib.model import Challenge
from collections.abc import MutableSequence

logger = logging.getLogger(__name__)

# Fields from the `matchmaking:` config section that can be overridden per slot.
_MATCHMAKING_KEYS = frozenset({
    "allow_matchmaking",
    "allow_during_games",
    "challenge_initial_time",
    "challenge_increment",
    "challenge_variant",
    "challenge_mode",
    "challenge_timeout",
    "challenge_filter",
    "opponent_min_rating",
    "opponent_max_rating",
    "opponent_rating_difference",
    "rating_preference",
    "block_list",
    "online_block_list",
})

# Fields from the `challenge:` config section that can be overridden per slot
# to filter which incoming challenges are assigned to that slot.
_ACCEPT_KEYS = frozenset({
    "accept_bot",
    "only_bot",
    "time_controls",
    "variants",
    "modes",
    "min_rating",
    "max_rating",
    "rating_difference",
    "min_increment",
    "max_increment",
    "min_base",
    "max_base",
    "min_days",
    "max_days",
    "bullet_requires_increment",
})

# Fields from the `correspondence:` config section that feed into the
# synthetic matchmaking config for outbound correspondence challenges.
_CORRESPONDENCE_MM_KEYS = frozenset({
    "challenge_days",
    "challenge_variant",
    "challenge_mode",
    "challenge_timeout",
    "challenge_filter",
    "opponent_min_rating",
    "opponent_max_rating",
    "opponent_rating_difference",
    "rating_preference",
    "allow_during_games",
    "block_list",
    "online_block_list",
})


@dataclass
class SlotConfig:
    """Configuration for a single matchmaking slot."""
    index: int               # Position in config list (0-based). Lower = first pick for incoming.
    name: str                # Human-readable label for log output.
    accept_cfg: Configuration  # Merged: base challenge config | slot accept-field overrides.
    allow_matchmaking: bool  # Whether this slot sends outgoing challenges.
    # Correspondence check-in options:
    correspondence_allowed: bool   # Whether this slot can host correspondence check-ins.
    correspondence_eviction: str   # "none" | "play_best" | "requeue"


class SlotManager:
    """
    Manages per-slot concurrency for real-time games.

    In legacy mode (no ``matchmaking.slots`` in config), this class is a transparent
    wrapper around the original ``Matchmaking`` class with zero behavioral change.

    In slot mode, each slot has its own independent ``Matchmaking`` instance that
    fires outgoing challenges on its own timer. Config list order determines which
    slot gets first pick of matching incoming challenges.

    Correspondence check-ins can be assigned to designated slots (``correspondence_allowed``).
    When a real-time challenge arrives and preemption is configured, the correspondence
    engine is signalled to stop via a ``multiprocessing.Event``.
    """

    def __init__(self, li: Lichess, config: Configuration, user_profile: UserProfileType) -> None:
        """Initialise from config. Falls back to legacy Matchmaking if no slots are configured."""
        self.li = li
        self.config = config
        self.user_profile = user_profile

        raw_slots = config.matchmaking.slots  # list of slot dicts, or None
        self.use_slots: bool = bool(raw_slots)

        if not self.use_slots:
            # Legacy mode: single Matchmaking instance, no slot tracking.
            self.legacy_matchmaker = matchmaking_module.Matchmaking(li, config, user_profile)
        else:
            assert isinstance(raw_slots, list)

            self.slots: list[SlotConfig] = []
            self.slot_matchmakers: dict[int, matchmaking_module.Matchmaking] = {}
            # index -> game_id currently in that slot (None = free)
            self.slot_to_game: dict[int, str | None] = {}
            # game_id -> slot index
            self.game_to_slot: dict[str, int] = {}
            # game_id -> preempt event (correspondence check-ins only)
            self.preempt_events: dict[str, multiprocessing.synchronize.Event] = {}

            slot_idx = 0
            for raw in raw_slots:
                base_name = raw.get("name", f"slot_{slot_idx}")
                slot_concurrency = int(raw.get("concurrency", 1))
                allow_mm = bool(raw.get("allow_matchmaking", False))
                correspondence_allowed = bool(raw.get("correspondence_allowed", False))
                correspondence_eviction = str(raw.get("correspondence_eviction", "none"))

                # Build accept_cfg once per config entry; all expanded slots share it.
                slot_accept_overrides = {k: v for k, v in raw.items() if k in _ACCEPT_KEYS}
                accept_cfg: Configuration = config.challenge | slot_accept_overrides
                synthetic_config = self._build_slot_config(config, raw) if allow_mm else None

                for i in range(slot_concurrency):
                    name = f"{base_name}[{i}]" if slot_concurrency > 1 else base_name

                    sc = SlotConfig(
                        index=slot_idx,
                        name=name,
                        accept_cfg=accept_cfg,
                        allow_matchmaking=allow_mm,
                        correspondence_allowed=correspondence_allowed,
                        correspondence_eviction=correspondence_eviction,
                    )
                    self.slots.append(sc)
                    self.slot_to_game[slot_idx] = None

                    if allow_mm:
                        assert synthetic_config is not None
                        self.slot_matchmakers[slot_idx] = matchmaking_module.Matchmaking(
                            li, synthetic_config, user_profile
                        )
                        logger.info(f"Slot '{name}' (index {slot_idx}): matchmaking enabled.")
                    else:
                        logger.info(
                            f"Slot '{name}' (index {slot_idx}): accept-only, no outgoing challenges."
                        )

                    if correspondence_allowed:
                        logger.info(
                            f"Slot '{name}' (index {slot_idx}): correspondence check-ins allowed "
                            f"(eviction={correspondence_eviction})."
                        )

                    slot_idx += 1

        # Outbound correspondence matchmaking (both legacy and slot modes).
        # Active when correspondence.allow_matchmaking is True AND correspondence.concurrency > 0.
        max_correspondence_games: int = config.correspondence.concurrency or 0
        corr_allow_mm = bool(config.correspondence.allow_matchmaking or False)
        if corr_allow_mm and max_correspondence_games > 0:
            corr_config = self._build_correspondence_config(config)
            self.correspondence_matchmaker: matchmaking_module.Matchmaking | None = (
                matchmaking_module.Matchmaking(li, corr_config, user_profile)
            )
            logger.info("Correspondence outbound matchmaking enabled.")
        else:
            self.correspondence_matchmaker = None

    # -----------------------------------------------------------------------
    # Public interface — mirrors Matchmaking for call-site compatibility
    # -----------------------------------------------------------------------

    def show_earliest_challenge_time(self) -> None:
        """Log the earliest time a challenge can be created for each active matchmaker."""
        if not self.use_slots:
            self.legacy_matchmaker.show_earliest_challenge_time()
        else:
            for mm in self.slot_matchmakers.values():
                mm.show_earliest_challenge_time()
        if self.correspondence_matchmaker is not None:
            self.correspondence_matchmaker.show_earliest_challenge_time()

    def game_done(self, game_id: str = "") -> None:
        """
        Notify that a game (or outbound challenge) has ended.

        In legacy mode the ``game_id`` parameter is ignored to remain compatible
        with the upstream ``Matchmaking.game_done()`` signature (no args).
        """
        if not self.use_slots:
            self.legacy_matchmaker.game_done()
            return
        slot_index = self.game_to_slot.pop(game_id, None)
        if slot_index is not None:
            self.slot_to_game[slot_index] = None
            self.preempt_events.pop(game_id, None)  # Clean up any preemption event.
            logger.info(f"Slot '{self.slots[slot_index].name}' freed (game {game_id}).")
            if slot_index in self.slot_matchmakers:
                self.slot_matchmakers[slot_index].game_done()

    def correspondence_game_done(self) -> None:
        """
        Notify the correspondence matchmaker that a correspondence game ended.

        Called from ``lichess_bot_main`` when ``local_game_done`` fires for a game
        that was in ``active_correspondence_game_ids``.  The correspondence matchmaker
        resets its delay timer so it can issue another outbound challenge sooner.
        """
        if self.correspondence_matchmaker is not None:
            self.correspondence_matchmaker.game_done()

    def accepted_challenge(self, event: EventType) -> None:
        """
        Notify that the opponent accepted a challenge (inbound or outbound).

        For inbound challenges the slot is already assigned in ``assign_game_to_slot()``.
        For outbound challenges we identify the issuing slot by matching ``challenge_id``.
        """
        if not self.use_slots:
            self.legacy_matchmaker.accepted_challenge(event)
            return
        game_id = event["game"]["id"]
        slot_index = self.game_to_slot.get(game_id)
        if slot_index is None:
            # Outbound challenge: find the slot matchmaker that sent it.
            for idx, mm in self.slot_matchmakers.items():
                if mm.challenge_id == game_id:
                    self.slot_to_game[idx] = game_id
                    self.game_to_slot[game_id] = idx
                    slot_index = idx
                    break
            # Check if it was the correspondence matchmaker.
            if slot_index is None and self.correspondence_matchmaker is not None:
                if self.correspondence_matchmaker.challenge_id == game_id:
                    self.correspondence_matchmaker.accepted_challenge(event)
                    return
            if slot_index is None:
                logger.warning(
                    f"accepted_challenge: could not find slot for game {game_id}. "
                    "Game will run without slot tracking."
                )
        if slot_index is not None and slot_index in self.slot_matchmakers:
            self.slot_matchmakers[slot_index].accepted_challenge(event)

    def declined_challenge(self, event: EventType) -> None:
        """Route a declined-challenge event to the slot matchmaker that sent it."""
        if not self.use_slots:
            self.legacy_matchmaker.declined_challenge(event)
            return
        challenge_id = event["challenge"]["id"]
        for mm in self.slot_matchmakers.values():
            if mm.challenge_id == challenge_id:
                mm.declined_challenge(event)
                return
        # Check correspondence matchmaker.
        if (self.correspondence_matchmaker is not None
                and self.correspondence_matchmaker.challenge_id == challenge_id):
            self.correspondence_matchmaker.declined_challenge(event)

    def challenge(
        self,
        active_games: set[str],
        challenge_queue: MutableSequence[object],
        max_games: int,
    ) -> None:
        """
        Issue outgoing challenges for all free slots that have matchmaking enabled.

        Each slot fires independently on its own timer — no slot blocks another.
        We pass an empty active_games set and max_games=1 to each slot matchmaker
        so the existing Matchmaking capacity logic correctly sees "my slot is free".
        The actual slot-occupancy check is done here before calling the matchmaker.

        A global CPU cap is also applied: ``active_games`` contains all currently
        running engine processes, including correspondence check-ins that are
        temporarily using a core to calculate a move. If total active engines already
        equals ``max_games``, no new outgoing challenges are issued — regardless of
        which slots show as free in ``slot_to_game``.
        """
        if not self.use_slots:
            self.legacy_matchmaker.challenge(active_games, challenge_queue, max_games)
            return
        # Global CPU cap: active_games includes real-time games AND any correspondence
        # check-ins currently running. Don't issue challenges when at full capacity.
        if len(active_games) >= max_games:
            return
        for sc in self.slots:
            if not sc.allow_matchmaking:
                continue
            if self.slot_to_game.get(sc.index) is not None:
                continue  # Slot occupied; skip.
            mm = self.slot_matchmakers.get(sc.index)
            if mm is None:
                continue
            # Pass empty context so the matchmaker sees "0 games, capacity 1".
            # The slot-occupancy guard above is the real capacity check.
            mm.challenge(set(), [], 1)

    def challenge_correspondence(
        self,
        active_games: set[str],
        challenge_queue: MutableSequence[object],
        max_games: int,
        correspondence_total: int,
        max_correspondence_games: int,
    ) -> None:
        """
        Issue outbound correspondence challenges when under both CPU and game caps.

        ``correspondence_total`` is ``len(active_correspondence_game_ids) +
        correspondence_queue.qsize()`` — the total number of correspondence games
        currently tracked (active check-ins + games waiting for their next check-in).
        """
        if self.correspondence_matchmaker is None:
            return
        if len(active_games) >= max_games:
            return
        if correspondence_total >= max_correspondence_games:
            return
        self.correspondence_matchmaker.challenge(set(), [], 1)

    # -----------------------------------------------------------------------
    # Slot assignment methods (called from slot_accept_challenges in lichess_bot.py)
    # -----------------------------------------------------------------------

    def find_slot_for_challenge(self, challenge: Challenge) -> SlotConfig | None:
        """
        Return the first free slot (in config list order) that accepts this challenge.

        Only lightweight slot-differentiating checks are done here. Heavy validation
        (block lists, rating bounds, variant, etc.) was already performed by
        ``handle_challenge()`` before the challenge entered the queue.

        Returns ``None`` if no free slot accepts the challenge.
        """
        for sc in self.slots:
            if self.slot_to_game.get(sc.index) is not None:
                continue  # Slot occupied.
            if self._challenge_fits_slot(challenge, sc):
                return sc
        return None

    def find_evictable_slot_for_challenge(self, challenge: Challenge) -> SlotConfig | None:
        """
        Return the first slot that is occupied by an evictable correspondence check-in
        and would accept this (real-time) challenge.

        A slot is evictable when ``correspondence_eviction != "none"`` and the current
        occupant is a correspondence check-in (i.e. has an entry in ``preempt_events``).

        Returns ``None`` if no evictable match is found.
        """
        if not self.use_slots:
            return None
        for sc in self.slots:
            if sc.correspondence_eviction == "none":
                continue  # Not configured for eviction.
            game_id = self.slot_to_game.get(sc.index)
            if game_id is None:
                continue  # Free slot — handled by find_slot_for_challenge.
            if game_id not in self.preempt_events:
                continue  # Occupied by a real-time game, not a correspondence check-in.
            if self._challenge_fits_slot(challenge, sc):
                return sc
        return None

    def assign_game_to_slot(self, game_id: str, slot_index: int) -> None:
        """Record that ``game_id`` now occupies the given slot."""
        self.slot_to_game[slot_index] = game_id
        self.game_to_slot[game_id] = slot_index
        logger.info(f"Game {game_id} assigned to slot '{self.slots[slot_index].name}'.")

    # -----------------------------------------------------------------------
    # Correspondence slot methods (called from check_in_on_correspondence_games)
    # -----------------------------------------------------------------------

    @property
    def has_correspondence_slots(self) -> bool:
        """Whether any slot is designated for correspondence check-ins."""
        return self.use_slots and any(sc.correspondence_allowed for sc in self.slots)

    def find_slot_for_correspondence_checkin(self) -> SlotConfig | None:
        """
        Return the first free slot with ``correspondence_allowed = True``.

        Returns ``None`` if all correspondence slots are occupied.
        """
        for sc in self.slots:
            if not sc.correspondence_allowed:
                continue
            if self.slot_to_game.get(sc.index) is not None:
                continue  # Occupied.
            return sc
        return None

    def register_correspondence_checkin(
        self,
        game_id: str,
        slot_index: int,
        event: multiprocessing.synchronize.Event,
    ) -> None:
        """
        Record that a correspondence check-in occupies ``slot_index`` and store
        the preemption event so it can be signalled later.
        """
        self.slot_to_game[slot_index] = game_id
        self.game_to_slot[game_id] = slot_index
        self.preempt_events[game_id] = event
        logger.info(
            f"Correspondence check-in {game_id} assigned to slot "
            f"'{self.slots[slot_index].name}' (eviction={self.slots[slot_index].correspondence_eviction})."
        )

    def preempt_correspondence(self, slot_index: int) -> None:
        """
        Signal the correspondence game in ``slot_index`` to stop.

        Sets the ``multiprocessing.Event`` that was registered via
        ``register_correspondence_checkin()``. The ``play_game`` subprocess checks
        this event and exits at the next natural control point in the game loop.
        The slot is NOT freed here — it is freed when ``local_game_done`` fires.
        """
        game_id = self.slot_to_game.get(slot_index)
        if game_id is None:
            return
        event = self.preempt_events.get(game_id)
        if event is not None:
            event.set()
            logger.info(
                f"Preemption signal sent to correspondence game {game_id} "
                f"in slot '{self.slots[slot_index].name}'."
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _challenge_fits_slot(self, challenge: Challenge, sc: SlotConfig) -> bool:
        """
        Return True if this challenge satisfies the slot's acceptance constraints.

        Checks:
        - ``accept_bot`` / ``only_bot`` — bot vs. human filter
        - ``time_controls`` — if the slot lists specific speeds, the challenge speed must match
        """
        accept_cfg = sc.accept_cfg

        # Bot / human filter
        if not accept_cfg.accept_bot and challenge.challenger.is_bot:
            return False
        if accept_cfg.only_bot and not challenge.challenger.is_bot:
            return False

        # Time control filter (only applied when the slot overrides time_controls)
        slot_time_controls = accept_cfg.config.get("time_controls")
        if slot_time_controls is not None:
            if challenge.speed not in slot_time_controls:
                return False

        return True

    @staticmethod
    def _build_slot_config(base_config: Configuration, raw_slot: dict) -> Configuration:
        """
        Build a synthetic full ``Configuration`` for a slot's ``Matchmaking`` instance.

        Starts from the base matchmaking config, applies slot-specific overrides,
        then injects it back into a copy of the full config dict.

        ``challenge_days`` is always set to ``[None]`` to ensure correspondence
        games are never selected by slot matchmakers (they are managed separately).
        """
        base_mm: dict = dict(base_config.matchmaking.config)
        slot_mm_overrides = {k: v for k, v in raw_slot.items() if k in _MATCHMAKING_KEYS}
        base_mm.update(slot_mm_overrides)

        # Normalize list fields (mirrors insert_default_values logic)
        for key in ("challenge_initial_time", "challenge_increment"):
            if key in base_mm and not isinstance(base_mm[key], list):
                base_mm[key] = [base_mm[key]]

        # Disable correspondence entirely in slot matchmakers
        base_mm["challenge_days"] = [None]

        # Enforce minimum challenge_timeout
        timeout = base_mm.get("challenge_timeout")
        if timeout is not None:
            base_mm["challenge_timeout"] = max(int(timeout), 1)

        # Build full synthetic config with this slot's matchmaking section
        synthetic_dict = dict(base_config.config)
        synthetic_dict["matchmaking"] = base_mm
        return Configuration(synthetic_dict)

    @staticmethod
    def _build_correspondence_config(base_config: Configuration) -> Configuration:
        """
        Build a synthetic ``Configuration`` for the correspondence ``Matchmaking`` instance.

        Starts from the base matchmaking config, then applies correspondence-specific
        overrides from the ``correspondence:`` config section.  ``challenge_initial_time``
        and ``challenge_increment`` are forced to ``[None]`` so the matchmaker always
        selects correspondence time controls (never real-time).
        """
        base_mm: dict = dict(base_config.matchmaking.config)
        corr_raw: dict = base_config.correspondence.config or {}

        corr_overrides: dict = {
            "allow_matchmaking": True,
            "challenge_initial_time": [None],  # No clock; correspondence only.
            "challenge_increment": [None],
        }
        for key in _CORRESPONDENCE_MM_KEYS:
            if key in corr_raw:
                corr_overrides[key] = corr_raw[key]

        base_mm.update(corr_overrides)

        # Ensure challenge_days is a list.
        days = base_mm.get("challenge_days")
        if days is not None and not isinstance(days, list):
            base_mm["challenge_days"] = [days]

        # Enforce minimum challenge_timeout.
        timeout = base_mm.get("challenge_timeout")
        if timeout is not None:
            base_mm["challenge_timeout"] = max(int(timeout), 1)

        synthetic_dict = dict(base_config.config)
        synthetic_dict["matchmaking"] = base_mm
        return Configuration(synthetic_dict)

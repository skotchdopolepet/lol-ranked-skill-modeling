from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import zstandard as zstd
except Exception:  # pragma: no cover
    zstd = None


VALID_TEAM_POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SORTED_VALID_TEAM_POSITIONS = sorted(VALID_TEAM_POSITIONS)
NORMALIZED_POSITION_NAMES = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "middle",
    "BOTTOM": "bottom",
    "UTILITY": "support",
}
POSITION_SET = set(VALID_TEAM_POSITIONS)

PARTICIPANT_NUMERIC_FEATURES = [
    "gold_pm", "xp_pm", "lane_cs_pm", "jungle_cs_pm", "kills_pm", "deaths_pm", "assists_pm",
    "dmg_to_champs_pm", "dmg_taken_pm", "vision_pm", "heal_pm", "damage_mitigated_pm",
    "total_time_spent_dead_pm",
    "wards_placed_pm", "wards_killed_pm", "control_wards_pm", "cc_time_pm",
    "damage_to_objectives_pm", "damage_to_turrets_pm", "turret_takedowns_pm", "dragon_kills_pm",
    "enemy_missing_pings_pm", "need_vision_pings_pm", "total_pings_pm",
]
PARTICIPANT_SHARE_FEATURES = [
    "dmg_taken_share", "kp_share", "damage_share", "vision_share", "gold_share", "control_wards_share",
]
PARTICIPANT_JUNGLE_ONLY_FEATURES = ["objectives_stolen_pm", "enemy_jungle_cs_pm"]
PARTICIPANT_DELTA_ONLY_FEATURES = ["first_blood", "first_tower"]

CHALLENGE_JUNGLE_FEATURES = [
    {"name": "jungle_scuttle_crab_kills_pm", "source": "scuttleCrabKills", "normalize_pm": True},
    {"name": "jungle_jungler_kills_early_jungle", "source": "junglerKillsEarlyJungle", "normalize_pm": False},
    {"name": "jungle_kills_on_laners_early_jungle_as_jungler", "source": "killsOnLanersEarlyJungleAsJungler", "normalize_pm": False},
    {"name": "jungle_cs_before_10_minutes", "source": "jungleCsBefore10Minutes", "normalize_pm": False},
    {"name": "jungle_initial_buff_count", "source": "initialBuffCount", "normalize_pm": False},
    {"name": "jungle_initial_crab_count", "source": "initialCrabCount", "normalize_pm": False},
    {"name": "jungle_epic_monster_kills_near_enemy_jungler_pm", "source": "epicMonsterKillsNearEnemyJungler", "normalize_pm": True},
    {"name": "jungle_epic_monster_kills_within_30_seconds_of_spawn_pm", "source": "epicMonsterKillsWithin30SecondsOfSpawn", "normalize_pm": True},
]
CHALLENGE_POSITION_FEATURES = [
    {"name": "teleport_takedowns_pm", "source": "teleportTakedowns", "positions": {"TOP", "MIDDLE"}, "normalize_pm": True},
    {"name": "lane_minions_first_10_minutes", "source": "laneMinionsFirst10Minutes", "positions": {"TOP", "MIDDLE", "BOTTOM"}, "normalize_pm": False},
    {"name": "turret_plates_taken", "source": "turretPlatesTaken", "positions": POSITION_SET, "normalize_pm": False},
    {"name": "k_turrets_destroyed_before_plates_fall", "source": "kTurretsDestroyedBeforePlatesFall", "positions": POSITION_SET, "normalize_pm": False},
    {"name": "kills_near_enemy_turret_pm", "source": "killsNearEnemyTurret", "positions": POSITION_SET, "normalize_pm": True},
    {"name": "enemy_champion_immobilizations_pm", "source": "enemyChampionImmobilizations", "positions": POSITION_SET, "normalize_pm": True},
    {"name": "effective_heal_and_shielding_pm", "source": "effectiveHealAndShielding", "positions": POSITION_SET, "normalize_pm": True},
    {"name": "solo_kills_pm", "source": "soloKills", "positions": POSITION_SET, "normalize_pm": True},
    {"name": "kills_under_own_turret_pm", "source": "killsUnderOwnTurret", "positions": POSITION_SET, "normalize_pm": True},
    {"name": "wards_guarded_pm", "source": "wardsGuarded", "positions": POSITION_SET, "normalize_pm": True},
    {"name": "control_ward_time_coverage_in_river_or_enemy_half", "source": "controlWardTimeCoverageInRiverOrEnemyHalf", "positions": POSITION_SET, "normalize_pm": False},
    {"name": "two_wards_one_sweeper_count_pm", "source": "twoWardsOneSweeperCount", "positions": {"JUNGLE", "UTILITY"}, "normalize_pm": True},
    {"name": "baron_takedowns", "source": "baronTakedowns", "positions": POSITION_SET, "normalize_pm": False},
    {"name": "dragon_takedowns_pm", "source": "dragonTakedowns", "positions": POSITION_SET, "normalize_pm": True},
]
CHALLENGE_DELTA_ONLY_FEATURES = [
    {"name": "rift_herald_takedown", "source": "riftHeraldTakedowns", "positions": POSITION_SET},
]

TEAM_CONTEXT_COLUMNS = [
    "perfect_dragon_souls_taken_delta", "team_elder_dragon_kills_avg", "team_elder_dragon_kills_delta",
    "first_turret_killed_time", "winner_earliest_baron", "loser_earliest_baron",
    "winner_earliest_dragon_takedown", "loser_earliest_dragon_takedown",
    "winner_earliest_elder_dragon", "loser_earliest_elder_dragon", "turrets_taken_with_rift_herald_delta",
    "winner_had_open_nexus", "winner_lost_an_inhibitor", "winner_fountain_takedowns_sum",
]
OBJECTIVES_DTO_COLUMNS = [
    "team_baron_kills_avg", "team_baron_kills_delta", "first_baron_delta",
    "team_dragon_kills_avg", "team_dragon_kills_delta", "team_dragon_kills_pm_avg", "team_dragon_kills_pm_delta", "first_dragon_delta",
    "rift_herald_kills_delta",
    "team_tower_kills_avg", "team_tower_kills_delta", "first_tower_delta",
    "team_inhibitor_kills_avg", "team_inhibitor_kills_delta", "first_inhibitor_delta",
    "team_grubs_kills_avg", "team_grubs_kills_delta", "first_grubs_delta",
]
SOCIAL_CONTEXT_COLUMNS = ["fist_bump_participation_match_avg"]
TOTAL_PING_FIELDS = [
    "allInPings", "assistMePings", "basicPings", "commandPings", "dangerPings", "enemyMissingPings",
    "enemyVisionPings", "getBackPings", "holdPings", "needVisionPings", "onMyWayPings", "pushPings",
    "retreatPings", "visionClearedPings",
]
METADATA_COLUMNS = [
    ("match_id", "TEXT PRIMARY KEY"), ("source_folder", "TEXT NOT NULL"), ("source_file", "TEXT NOT NULL"),
    ("game_version", "TEXT NOT NULL"), ("patch", "TEXT NOT NULL"), ("queue_id", "INTEGER NOT NULL"),
    ("map_id", "INTEGER NOT NULL"), ("server_timezone_fallback_used", "INTEGER NOT NULL"),
]
INFO_VARIABLE_COLUMNS = [("game_duration_sec", "REAL NOT NULL")]
SECONDARY_ANALYSIS_COLUMNS = [
    ("server", "TEXT NOT NULL"),
    ("is_weekend", "INTEGER NOT NULL"),
    ("local_time_sin", "REAL NOT NULL"),
    ("local_time_cos", "REAL NOT NULL"),
]
SERVER_TIMEZONE_MAP = {
    "BR1": "America/Sao_Paulo",
    "EUN1": "Europe/Warsaw",
    "EUW1": "Europe/Amsterdam",
    "JP1": "Asia/Tokyo",
    "KR": "Asia/Seoul",
    "LA1": "America/Mexico_City",
    "LA2": "America/Santiago",
    "ME1": "Asia/Dubai",
    "NA1": "America/Chicago",
    "OC1": "Australia/Sydney",
    "PBE1": "America/Los_Angeles",
    "PH2": "Asia/Manila",
    "RU": "Europe/Moscow",
    "SG2": "Asia/Singapore",
    "TH2": "Asia/Bangkok",
    "TR1": "Europe/Istanbul",
    "TW2": "Asia/Taipei",
    "VN2": "Asia/Ho_Chi_Minh",
}
SERVER_TIMEZONE_FAMILY_MAP = {
    "BR": "America/Sao_Paulo",
    "EUN": "Europe/Warsaw",
    "EUW": "Europe/Amsterdam",
    "JP": "Asia/Tokyo",
    "KR": "Asia/Seoul",
    "LA": "America/Mexico_City",
    "ME": "Asia/Dubai",
    "NA": "America/Chicago",
    "OC": "Australia/Sydney",
    "PBE": "America/Los_Angeles",
    "PH": "Asia/Manila",
    "RU": "Europe/Moscow",
    "SG": "Asia/Singapore",
    "TH": "Asia/Bangkok",
    "TR": "Europe/Istanbul",
    "TW": "Asia/Taipei",
    "VN": "Asia/Ho_Chi_Minh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a match-level feature table from match JSONs using teamPosition only."
    )
    parser.add_argument("--input-root", default="runtime/out_prod")
    parser.add_argument("--output-dir", default="runtime/out_latest/datasets/features/match_feature_table_v1")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--sample-random", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def iter_match_files(input_root: Path):
    source_dirs = [input_root] if (input_root / "matches").exists() else [c for c in sorted(input_root.iterdir()) if c.is_dir()]
    for child in source_dirs:
        matches_dir = child / "matches"
        if not matches_dir.exists():
            continue
        for match_path in sorted(matches_dir.iterdir()):
            if match_path.is_file() and (match_path.name.endswith(".json") or match_path.name.endswith(".json.zst")):
                yield child.name, match_path


def load_payload(path: Path) -> dict[str, Any] | None:
    try:
        if path.name.endswith(".json.zst"):
            if zstd is None:
                return None
            with path.open("rb") as f:
                raw = zstd.ZstdDecompressor().stream_reader(f).read()
            obj = json.loads(raw.decode("utf-8"))
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def patch_from_version(game_version: str) -> str:
    parts = str(game_version or "").split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(game_version or "")


def server_from_match_id(match_id: str) -> str:
    token = str(match_id or "")
    return token.split("_", 1)[0] if "_" in token else token


def timezone_for_server(server: str, source_folder: str) -> tuple[str, int]:
    server_token = str(server or "").upper()
    folder_token = str(source_folder or "").lower()

    exact = SERVER_TIMEZONE_MAP.get(server_token)
    if exact:
        return exact, 0

    folder_rules = [
        ("euw", "Europe/Amsterdam"),
        ("eune", "Europe/Warsaw"),
        ("na", "America/Chicago"),
        ("br", "America/Sao_Paulo"),
        ("lan", "America/Mexico_City"),
        ("las", "America/Santiago"),
        ("kr", "Asia/Seoul"),
        ("jp", "Asia/Tokyo"),
        ("tw", "Asia/Taipei"),
        ("sg", "Asia/Singapore"),
        ("ph", "Asia/Manila"),
        ("th", "Asia/Bangkok"),
        ("vn", "Asia/Ho_Chi_Minh"),
        ("oce", "Australia/Sydney"),
        ("oc", "Australia/Sydney"),
        ("tr", "Europe/Istanbul"),
        ("ru", "Europe/Moscow"),
        ("me", "Asia/Dubai"),
        ("pbe", "America/Los_Angeles"),
    ]
    for needle, zone_name in folder_rules:
        if needle in folder_token:
            return zone_name, 1

    family_matches = sorted(
        ((family, zone_name) for family, zone_name in SERVER_TIMEZONE_FAMILY_MAP.items() if server_token.startswith(family)),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    if family_matches:
        return family_matches[0][1], 1

    return "UTC", 1


def duration_seconds(info: dict[str, Any]) -> float:
    duration_sec = float(info.get("gameDuration") or 0.0)
    if duration_sec <= 0:
        start_ts = int(info.get("gameStartTimestamp") or 0)
        end_ts = int(info.get("gameEndTimestamp") or 0)
        if end_ts > start_ts > 0:
            duration_sec = float((end_ts - start_ts) / 1000.0)
    return float(duration_sec) if duration_sec > 0 else 0.0


def duration_minutes(info: dict[str, Any]) -> float:
    duration_sec = duration_seconds(info)
    return float(duration_sec / 60.0) if duration_sec > 0 else 0.0


def local_time_features(game_start_timestamp_ms: int, server: str, source_folder: str) -> dict[str, float | int]:
    if int(game_start_timestamp_ms or 0) <= 0:
        return {"is_weekend": 0, "local_time_sin": 0.0, "local_time_cos": 1.0, "server_timezone_fallback_used": 1}
    zone_name, fallback_used = timezone_for_server(server, source_folder)
    utc_dt = datetime.fromtimestamp(float(game_start_timestamp_ms) / 1000.0, tz=timezone.utc)
    try:
        local_dt = utc_dt.astimezone(ZoneInfo(zone_name))
    except Exception:
        local_dt = utc_dt
    local_seconds_since_midnight = (local_dt.hour * 3600) + (local_dt.minute * 60) + local_dt.second
    angle = 2.0 * math.pi * (float(local_seconds_since_midnight) / 86400.0)
    return {
        "is_weekend": 1 if local_dt.weekday() >= 5 else 0,
        "local_time_sin": float(math.sin(angle)),
        "local_time_cos": float(math.cos(angle)),
        "server_timezone_fallback_used": int(fallback_used),
    }


def valid_position(pos: str) -> bool:
    return str(pos or "") in VALID_TEAM_POSITIONS


def build_position_map(participants: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]] | None:
    by_team: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for part in participants:
        team_id = int(part.get("teamId") or 0)
        team_position = str(part.get("teamPosition") or "")
        if not valid_position(team_position) or team_position in by_team[team_id]:
            return None
        by_team[team_id][team_position] = part
    if sorted(by_team.keys()) != [100, 200]:
        return None
    for team_id in [100, 200]:
        if sorted(by_team[team_id].keys()) != SORTED_VALID_TEAM_POSITIONS:
            return None
    return by_team


def winner_loser_team_ids(participants: list[dict[str, Any]]) -> tuple[int, int] | None:
    flags_by_team: dict[int, set[bool]] = defaultdict(set)
    for part in participants:
        flags_by_team[int(part.get("teamId") or 0)].add(bool(part.get("win")))
    if sorted(flags_by_team.keys()) != [100, 200] or any(len(v) != 1 for v in flags_by_team.values()):
        return None
    team_100_win = next(iter(flags_by_team[100]))
    team_200_win = next(iter(flags_by_team[200]))
    if team_100_win == team_200_win:
        return None
    return (100, 200) if team_100_win else (200, 100)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num / den)


def team_participant_sum(team_parts: dict[str, dict[str, Any]], key: str) -> float:
    return float(sum(float(p.get(key) or 0.0) for p in team_parts.values()))


def participant_metrics(
    part: dict[str, Any],
    *,
    duration_min: float,
    team_kills: float,
    team_damage_to_champs: float,
    team_damage_taken: float,
    team_vision_score: float,
    team_gold: float,
    team_control_wards: float,
) -> dict[str, float]:
    kills = float(part.get("kills") or 0.0)
    deaths = float(part.get("deaths") or 0.0)
    assists = float(part.get("assists") or 0.0)
    gold_earned = float(part.get("goldEarned") or 0.0)
    champ_experience = float(part.get("champExperience") or 0.0)
    total_minions_killed = float(part.get("totalMinionsKilled") or 0.0)
    neutral_minions_killed = float(part.get("neutralMinionsKilled") or 0.0)
    damage_to_champs = float(part.get("totalDamageDealtToChampions") or 0.0)
    damage_taken = float(part.get("totalDamageTaken") or 0.0)
    damage_mitigated = float(part.get("damageSelfMitigated") or 0.0)
    damage_to_objectives = float(part.get("damageDealtToObjectives") or 0.0)
    damage_to_turrets = float(part.get("damageDealtToTurrets") or 0.0)
    vision_score = float(part.get("visionScore") or 0.0)
    wards_placed = float(part.get("wardsPlaced") or 0.0)
    wards_killed = float(part.get("wardsKilled") or 0.0)
    control_wards = float(part.get("detectorWardsPlaced") or 0.0)
    total_heal = float(part.get("totalHeal") or 0.0)
    total_time_spent_dead = float(part.get("totalTimeSpentDead") or 0.0)
    cc_time = float(part.get("timeCCingOthers") or 0.0)
    turret_takedowns = float(part.get("turretTakedowns") or 0.0)
    dragon_kills = float(part.get("dragonKills") or 0.0)
    objectives_stolen = float(part.get("objectivesStolen") or 0.0)
    enemy_jungle_cs = float(part.get("totalEnemyJungleMinionsKilled") or 0.0)
    enemy_missing_pings = float(part.get("enemyMissingPings") or 0.0)
    need_vision_pings = float(part.get("needVisionPings") or 0.0)
    total_pings = float(sum(float(part.get(name) or 0.0) for name in TOTAL_PING_FIELDS))
    return {
        "gold_pm": safe_div(gold_earned, duration_min),
        "xp_pm": safe_div(champ_experience, duration_min),
        "lane_cs_pm": safe_div(total_minions_killed, duration_min),
        "jungle_cs_pm": safe_div(neutral_minions_killed, duration_min),
        "kills_pm": safe_div(kills, duration_min),
        "deaths_pm": safe_div(deaths, duration_min),
        "assists_pm": safe_div(assists, duration_min),
        "dmg_to_champs_pm": safe_div(damage_to_champs, duration_min),
        "dmg_taken_pm": safe_div(damage_taken, duration_min),
        "vision_pm": safe_div(vision_score, duration_min),
        "heal_pm": safe_div(total_heal, duration_min),
        "damage_mitigated_pm": safe_div(damage_mitigated, duration_min),
        "total_time_spent_dead_pm": safe_div(total_time_spent_dead, duration_min),
        "wards_placed_pm": safe_div(wards_placed, duration_min),
        "wards_killed_pm": safe_div(wards_killed, duration_min),
        "control_wards_pm": safe_div(control_wards, duration_min),
        "cc_time_pm": safe_div(cc_time, duration_min),
        "damage_to_objectives_pm": safe_div(damage_to_objectives, duration_min),
        "damage_to_turrets_pm": safe_div(damage_to_turrets, duration_min),
        "turret_takedowns_pm": safe_div(turret_takedowns, duration_min),
        "dragon_kills_pm": safe_div(dragon_kills, duration_min),
        "enemy_missing_pings_pm": safe_div(enemy_missing_pings, duration_min),
        "need_vision_pings_pm": safe_div(need_vision_pings, duration_min),
        "total_pings_pm": safe_div(total_pings, duration_min),
        "dmg_taken_share": safe_div(damage_taken, team_damage_taken),
        "kp_share": safe_div(kills + assists, team_kills),
        "damage_share": safe_div(damage_to_champs, team_damage_to_champs),
        "vision_share": safe_div(vision_score, team_vision_score),
        "gold_share": safe_div(gold_earned, team_gold),
        "control_wards_share": safe_div(control_wards, team_control_wards),
        "objectives_stolen_pm": safe_div(objectives_stolen, duration_min),
        "enemy_jungle_cs_pm": safe_div(enemy_jungle_cs, duration_min),
    }


def participant_delta_flags(part: dict[str, Any]) -> dict[str, float]:
    first_blood = bool(part.get("firstBloodKill")) or bool(part.get("firstBloodAssist"))
    first_tower = bool(part.get("firstTowerKill")) or bool(part.get("firstTowerAssist"))
    return {"first_blood": 1.0 if first_blood else 0.0, "first_tower": 1.0 if first_tower else 0.0}


def challenge_numeric_value(part: dict[str, Any], source: str) -> float:
    return float(((part.get("challenges") or {}).get(source) or 0.0))


def challenge_feature_value(part: dict[str, Any], source: str, duration_min: float, normalize_pm: bool) -> float:
    value = challenge_numeric_value(part, source)
    return safe_div(value, duration_min) if normalize_pm else value


def challenge_binary_flag(part: dict[str, Any], source: str) -> float:
    return 1.0 if challenge_numeric_value(part, source) > 0 else 0.0


def max_team_challenge_value(team_parts: dict[str, dict[str, Any]], source: str) -> float:
    return max((challenge_numeric_value(part, source) for part in team_parts.values()), default=0.0)


def min_positive_team_challenge_value(team_parts: dict[str, dict[str, Any]], source: str) -> float:
    positives = [challenge_numeric_value(part, source) for part in team_parts.values() if challenge_numeric_value(part, source) > 0]
    return min(positives) if positives else 0.0


def sum_team_challenge_value(team_parts: dict[str, dict[str, Any]], source: str) -> float:
    return float(sum(challenge_numeric_value(part, source) for part in team_parts.values()))


def average_match_challenge_value(participants: list[dict[str, Any]], source: str) -> float:
    return 0.0 if not participants else float(sum(challenge_numeric_value(part, source) for part in participants) / len(participants))


def build_team_map(info: dict[str, Any]) -> dict[int, dict[str, Any]] | None:
    teams = info.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        return None
    by_team = {int(team.get("teamId") or 0): team for team in teams if isinstance(team, dict)}
    return by_team if set(by_team.keys()) == {100, 200} else None


def objective_kills(team_payload: dict[str, Any], objective_name: str) -> float:
    objectives = team_payload.get("objectives") or {}
    objective = objectives.get(objective_name) or {}
    return float(objective.get("kills") or 0.0)


def objective_first(team_payload: dict[str, Any], objective_name: str) -> float:
    objectives = team_payload.get("objectives") or {}
    objective = objectives.get(objective_name) or {}
    return 1.0 if bool(objective.get("first")) else 0.0


def schema_feature_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = list(METADATA_COLUMNS)
    columns.extend(INFO_VARIABLE_COLUMNS)
    columns.extend(SECONDARY_ANALYSIS_COLUMNS)
    for api_pos in VALID_TEAM_POSITIONS:
        pos_name = NORMALIZED_POSITION_NAMES[api_pos]
        for feature_name in PARTICIPANT_NUMERIC_FEATURES:
            columns.append((f"{pos_name}_{feature_name}_avg", "REAL NOT NULL"))
            columns.append((f"{pos_name}_{feature_name}_delta", "REAL NOT NULL"))
        for feature_name in PARTICIPANT_SHARE_FEATURES:
            columns.append((f"{pos_name}_{feature_name}_avg", "REAL NOT NULL"))
        if api_pos == "JUNGLE":
            for feature_name in PARTICIPANT_JUNGLE_ONLY_FEATURES:
                columns.append((f"{pos_name}_{feature_name}_avg", "REAL NOT NULL"))
                columns.append((f"{pos_name}_{feature_name}_delta", "REAL NOT NULL"))
        for feature_name in PARTICIPANT_DELTA_ONLY_FEATURES:
            columns.append((f"{pos_name}_{feature_name}_delta", "REAL NOT NULL"))
        for spec in CHALLENGE_POSITION_FEATURES:
            if api_pos in spec["positions"]:
                columns.append((f"{pos_name}_{spec['name']}_avg", "REAL NOT NULL"))
                columns.append((f"{pos_name}_{spec['name']}_delta", "REAL NOT NULL"))
        if api_pos == "JUNGLE":
            for spec in CHALLENGE_JUNGLE_FEATURES:
                columns.append((f"{pos_name}_{spec['name']}_avg", "REAL NOT NULL"))
                columns.append((f"{pos_name}_{spec['name']}_delta", "REAL NOT NULL"))
        for spec in CHALLENGE_DELTA_ONLY_FEATURES:
            if api_pos in spec["positions"]:
                columns.append((f"{pos_name}_{spec['name']}_delta", "REAL NOT NULL"))
    for feature_name in TEAM_CONTEXT_COLUMNS:
        columns.append((feature_name, "REAL NOT NULL"))
    for feature_name in OBJECTIVES_DTO_COLUMNS:
        columns.append((feature_name, "REAL NOT NULL"))
    for feature_name in SOCIAL_CONTEXT_COLUMNS:
        columns.append((feature_name, "REAL NOT NULL"))
    return columns


def create_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    columns_sql = ",\n".join(f"{name} {ctype}" for name, ctype in schema_feature_columns())
    conn.execute(f"CREATE TABLE match_table_v1 (\n{columns_sql}\n)")
    conn.execute("CREATE INDEX idx_match_table_v1_patch ON match_table_v1(patch)")
    conn.execute("CREATE INDEX idx_match_table_v1_queue_id ON match_table_v1(queue_id)")
    conn.execute("CREATE INDEX idx_match_table_v1_source_folder ON match_table_v1(source_folder)")
    conn.commit()
    return conn


def insert_row(conn: sqlite3.Connection, row: dict[str, Any]) -> bool:
    columns = list(row.keys())
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT OR IGNORE INTO match_table_v1 ({', '.join(columns)}) VALUES ({placeholders})"
    cur = conn.execute(sql, [row[c] for c in columns])
    return bool(cur.rowcount)


def extract_row(*, source_folder: str, path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    info = payload.get("info")
    metadata = payload.get("metadata")
    if not isinstance(info, dict) or not isinstance(metadata, dict):
        return None, "missing_info_or_metadata"
    participants = info.get("participants")
    if not isinstance(participants, list) or len(participants) != 10:
        return None, "participant_count_not_10"
    team_map = build_team_map(info)
    if team_map is None:
        return None, "invalid_team_payload"

    pos_map = build_position_map(participants)
    if pos_map is None:
        return None, "invalid_team_positions"
    winner_loser = winner_loser_team_ids(participants)
    if winner_loser is None:
        return None, "invalid_win_flags"
    winner_team_id, loser_team_id = winner_loser

    match_id = str(metadata.get("matchId") or path.stem.replace(".json", ""))
    server = server_from_match_id(match_id)
    game_start_timestamp_ms = int(info.get("gameStartTimestamp") or 0)
    duration_sec = duration_seconds(info)
    duration_min = duration_minutes(info)
    if duration_sec <= 0 or duration_min <= 0:
        return None, "invalid_duration"

    winner_parts = pos_map[winner_team_id]
    loser_parts = pos_map[loser_team_id]
    winner_team_kills = team_participant_sum(winner_parts, "kills")
    loser_team_kills = team_participant_sum(loser_parts, "kills")
    winner_team_dmg = team_participant_sum(winner_parts, "totalDamageDealtToChampions")
    loser_team_dmg = team_participant_sum(loser_parts, "totalDamageDealtToChampions")
    winner_team_damage_taken = team_participant_sum(winner_parts, "totalDamageTaken")
    loser_team_damage_taken = team_participant_sum(loser_parts, "totalDamageTaken")
    winner_team_vision = team_participant_sum(winner_parts, "visionScore")
    loser_team_vision = team_participant_sum(loser_parts, "visionScore")
    winner_team_gold = team_participant_sum(winner_parts, "goldEarned")
    loser_team_gold = team_participant_sum(loser_parts, "goldEarned")
    winner_team_control_wards = team_participant_sum(winner_parts, "detectorWardsPlaced")
    loser_team_control_wards = team_participant_sum(loser_parts, "detectorWardsPlaced")
    local_time = local_time_features(game_start_timestamp_ms, server, source_folder)

    row: dict[str, Any] = {
        "match_id": match_id,
        "source_folder": source_folder,
        "source_file": path.name,
        "game_version": str(info.get("gameVersion") or ""),
        "patch": patch_from_version(str(info.get("gameVersion") or "")),
        "queue_id": int(info.get("queueId") or 0),
        "map_id": int(info.get("mapId") or 0),
        "server_timezone_fallback_used": int(local_time["server_timezone_fallback_used"]),
        "game_duration_sec": float(duration_sec),
        "server": server,
        "is_weekend": int(local_time["is_weekend"]),
        "local_time_sin": float(local_time["local_time_sin"]),
        "local_time_cos": float(local_time["local_time_cos"]),
    }

    for api_pos in VALID_TEAM_POSITIONS:
        pos_name = NORMALIZED_POSITION_NAMES[api_pos]
        winner_metrics = participant_metrics(
            winner_parts[api_pos],
            duration_min=duration_min,
            team_kills=winner_team_kills,
            team_damage_to_champs=winner_team_dmg,
            team_damage_taken=winner_team_damage_taken,
            team_vision_score=winner_team_vision,
            team_gold=winner_team_gold,
            team_control_wards=winner_team_control_wards,
        )
        loser_metrics = participant_metrics(
            loser_parts[api_pos],
            duration_min=duration_min,
            team_kills=loser_team_kills,
            team_damage_to_champs=loser_team_dmg,
            team_damage_taken=loser_team_damage_taken,
            team_vision_score=loser_team_vision,
            team_gold=loser_team_gold,
            team_control_wards=loser_team_control_wards,
        )
        for feature_name in PARTICIPANT_NUMERIC_FEATURES:
            winner_value = winner_metrics[feature_name]
            loser_value = loser_metrics[feature_name]
            row[f"{pos_name}_{feature_name}_avg"] = (winner_value + loser_value) / 2.0
            row[f"{pos_name}_{feature_name}_delta"] = winner_value - loser_value
        for feature_name in PARTICIPANT_SHARE_FEATURES:
            winner_value = winner_metrics[feature_name]
            loser_value = loser_metrics[feature_name]
            row[f"{pos_name}_{feature_name}_avg"] = (winner_value + loser_value) / 2.0
        if api_pos == "JUNGLE":
            for feature_name in PARTICIPANT_JUNGLE_ONLY_FEATURES:
                winner_value = winner_metrics[feature_name]
                loser_value = loser_metrics[feature_name]
                row[f"{pos_name}_{feature_name}_avg"] = (winner_value + loser_value) / 2.0
                row[f"{pos_name}_{feature_name}_delta"] = winner_value - loser_value
        winner_flags = participant_delta_flags(winner_parts[api_pos])
        loser_flags = participant_delta_flags(loser_parts[api_pos])
        for feature_name in PARTICIPANT_DELTA_ONLY_FEATURES:
            row[f"{pos_name}_{feature_name}_delta"] = winner_flags[feature_name] - loser_flags[feature_name]

        for spec in CHALLENGE_POSITION_FEATURES:
            if api_pos not in spec["positions"]:
                continue
            winner_value = challenge_feature_value(winner_parts[api_pos], spec["source"], duration_min, bool(spec["normalize_pm"]))
            loser_value = challenge_feature_value(loser_parts[api_pos], spec["source"], duration_min, bool(spec["normalize_pm"]))
            row[f"{pos_name}_{spec['name']}_avg"] = (winner_value + loser_value) / 2.0
            row[f"{pos_name}_{spec['name']}_delta"] = winner_value - loser_value
        if api_pos == "JUNGLE":
            for spec in CHALLENGE_JUNGLE_FEATURES:
                winner_value = challenge_feature_value(winner_parts[api_pos], spec["source"], duration_min, bool(spec["normalize_pm"]))
                loser_value = challenge_feature_value(loser_parts[api_pos], spec["source"], duration_min, bool(spec["normalize_pm"]))
                row[f"{pos_name}_{spec['name']}_avg"] = (winner_value + loser_value) / 2.0
                row[f"{pos_name}_{spec['name']}_delta"] = winner_value - loser_value
        for spec in CHALLENGE_DELTA_ONLY_FEATURES:
            if api_pos not in spec["positions"]:
                continue
            row[f"{pos_name}_{spec['name']}_delta"] = challenge_binary_flag(winner_parts[api_pos], spec["source"]) - challenge_binary_flag(loser_parts[api_pos], spec["source"])

    winner_perfect_dragon_souls = 1.0 if max_team_challenge_value(winner_parts, "perfectDragonSoulsTaken") > 0 else 0.0
    loser_perfect_dragon_souls = 1.0 if max_team_challenge_value(loser_parts, "perfectDragonSoulsTaken") > 0 else 0.0
    winner_team_elder = max_team_challenge_value(winner_parts, "teamElderDragonKills")
    loser_team_elder = max_team_challenge_value(loser_parts, "teamElderDragonKills")
    row["perfect_dragon_souls_taken_delta"] = winner_perfect_dragon_souls - loser_perfect_dragon_souls
    row["team_elder_dragon_kills_avg"] = (winner_team_elder + loser_team_elder) / 2.0
    row["team_elder_dragon_kills_delta"] = winner_team_elder - loser_team_elder
    row["first_turret_killed_time"] = max(max_team_challenge_value(winner_parts, "firstTurretKilledTime"), max_team_challenge_value(loser_parts, "firstTurretKilledTime"))
    row["winner_earliest_baron"] = max_team_challenge_value(winner_parts, "earliestBaron")
    row["loser_earliest_baron"] = max_team_challenge_value(loser_parts, "earliestBaron")
    row["winner_earliest_dragon_takedown"] = min_positive_team_challenge_value(winner_parts, "earliestDragonTakedown")
    row["loser_earliest_dragon_takedown"] = min_positive_team_challenge_value(loser_parts, "earliestDragonTakedown")
    row["winner_earliest_elder_dragon"] = max_team_challenge_value(winner_parts, "earliestElderDragon")
    row["loser_earliest_elder_dragon"] = max_team_challenge_value(loser_parts, "earliestElderDragon")
    row["turrets_taken_with_rift_herald_delta"] = max_team_challenge_value(winner_parts, "turretsTakenWithRiftHerald") / 5.0 - max_team_challenge_value(loser_parts, "turretsTakenWithRiftHerald") / 5.0
    row["winner_had_open_nexus"] = 1.0 if max_team_challenge_value(winner_parts, "hadOpenNexus") > 0 else 0.0
    row["winner_lost_an_inhibitor"] = 1.0 if max_team_challenge_value(winner_parts, "lostAnInhibitor") > 0 else 0.0
    row["winner_fountain_takedowns_sum"] = sum_team_challenge_value(winner_parts, "takedownsInEnemyFountain")

    winner_team_payload = team_map[winner_team_id]
    loser_team_payload = team_map[loser_team_id]
    winner_baron = objective_kills(winner_team_payload, "baron")
    loser_baron = objective_kills(loser_team_payload, "baron")
    winner_dragon = objective_kills(winner_team_payload, "dragon")
    loser_dragon = objective_kills(loser_team_payload, "dragon")
    winner_herald = objective_kills(winner_team_payload, "riftHerald")
    loser_herald = objective_kills(loser_team_payload, "riftHerald")
    winner_tower = objective_kills(winner_team_payload, "tower")
    loser_tower = objective_kills(loser_team_payload, "tower")
    winner_inhib = objective_kills(winner_team_payload, "inhibitor")
    loser_inhib = objective_kills(loser_team_payload, "inhibitor")
    winner_grubs = objective_kills(winner_team_payload, "horde")
    loser_grubs = objective_kills(loser_team_payload, "horde")

    row["team_baron_kills_avg"] = (winner_baron + loser_baron) / 2.0
    row["team_baron_kills_delta"] = winner_baron - loser_baron
    row["first_baron_delta"] = objective_first(winner_team_payload, "baron") - objective_first(loser_team_payload, "baron")

    row["team_dragon_kills_avg"] = (winner_dragon + loser_dragon) / 2.0
    row["team_dragon_kills_delta"] = winner_dragon - loser_dragon
    row["team_dragon_kills_pm_avg"] = safe_div(row["team_dragon_kills_avg"], duration_min)
    row["team_dragon_kills_pm_delta"] = safe_div(row["team_dragon_kills_delta"], duration_min)
    row["first_dragon_delta"] = objective_first(winner_team_payload, "dragon") - objective_first(loser_team_payload, "dragon")

    row["rift_herald_kills_delta"] = winner_herald - loser_herald

    row["team_tower_kills_avg"] = (winner_tower + loser_tower) / 2.0
    row["team_tower_kills_delta"] = winner_tower - loser_tower
    row["first_tower_delta"] = objective_first(winner_team_payload, "tower") - objective_first(loser_team_payload, "tower")

    row["team_inhibitor_kills_avg"] = (winner_inhib + loser_inhib) / 2.0
    row["team_inhibitor_kills_delta"] = winner_inhib - loser_inhib
    row["first_inhibitor_delta"] = objective_first(winner_team_payload, "inhibitor") - objective_first(loser_team_payload, "inhibitor")

    row["team_grubs_kills_avg"] = (winner_grubs + loser_grubs) / 2.0
    row["team_grubs_kills_delta"] = winner_grubs - loser_grubs
    row["first_grubs_delta"] = objective_first(winner_team_payload, "horde") - objective_first(loser_team_payload, "horde")

    row["fist_bump_participation_match_avg"] = average_match_challenge_value(participants, "fistBumpParticipation")
    return row, "ok"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_files = int(args.max_files)
    if bool(args.sample_random):
        files = list(iter_match_files(input_root))
        if max_files > 0 and len(files) > max_files:
            files = random.Random(int(args.sample_seed)).sample(files, max_files)
    else:
        files: list[tuple[str, Path]] = []
        for idx, item in enumerate(iter_match_files(input_root), start=1):
            files.append(item)
            if max_files > 0 and idx >= max_files:
                break

    db_path = output_dir / "match_feature_table_v1.sqlite3"
    conn = create_db(db_path)
    kept = duplicate_match_ids = loaded = 0
    drop_reasons: Counter[str] = Counter()
    folder_kept: Counter[str] = Counter()
    examples_by_reason: dict[str, list[str]] = defaultdict(list)
    try:
        for source_folder, path in files:
            payload = load_payload(path)
            if payload is None:
                drop_reasons["load_error"] += 1
                if len(examples_by_reason["load_error"]) < 5:
                    examples_by_reason["load_error"].append(str(path))
                continue
            loaded += 1
            row, status = extract_row(source_folder=source_folder, path=path, payload=payload)
            if row is None:
                drop_reasons[status] += 1
                if len(examples_by_reason[status]) < 5:
                    examples_by_reason[status].append(str(path))
                continue
            if insert_row(conn, row):
                kept += 1
                folder_kept[source_folder] += 1
            else:
                duplicate_match_ids += 1
        conn.commit()
    finally:
        conn.close()

    non_metadata_column_count = len(schema_feature_columns()) - len(METADATA_COLUMNS)
    primary_model_feature_count = (
        len(PARTICIPANT_NUMERIC_FEATURES) * 5 * 2
        + len(PARTICIPANT_SHARE_FEATURES) * 5
        + len(PARTICIPANT_JUNGLE_ONLY_FEATURES) * 2
        + len(PARTICIPANT_DELTA_ONLY_FEATURES) * 5
        + sum(len(spec["positions"]) * 2 for spec in CHALLENGE_POSITION_FEATURES)
        + len(CHALLENGE_JUNGLE_FEATURES) * 2
        + sum(len(spec["positions"]) for spec in CHALLENGE_DELTA_ONLY_FEATURES)
        + len(TEAM_CONTEXT_COLUMNS)
        + len(OBJECTIVES_DTO_COLUMNS)
        + len(SOCIAL_CONTEXT_COLUMNS)
        + len(INFO_VARIABLE_COLUMNS)
    )
    secondary_analysis_feature_count = len(SECONDARY_ANALYSIS_COLUMNS)
    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "db_path": str(db_path),
        "discovered_files": len(files),
        "loaded_payloads": loaded,
        "kept_matches": kept,
        "duplicate_match_ids_skipped": duplicate_match_ids,
        "dropped_matches": int(sum(drop_reasons.values())),
        "drop_reasons": dict(drop_reasons),
        "drop_reason_examples": {k: v for k, v in sorted(examples_by_reason.items())},
        "primary_model_feature_count": primary_model_feature_count,
        "secondary_analysis_feature_count": secondary_analysis_feature_count,
        "non_metadata_column_count": non_metadata_column_count,
        "total_column_count": len(schema_feature_columns()),
        "metadata_columns": [name for name, _ in METADATA_COLUMNS],
        "info_variable_columns": [name for name, _ in INFO_VARIABLE_COLUMNS],
        "secondary_analysis_columns": [name for name, _ in SECONDARY_ANALYSIS_COLUMNS],
        "positions": [NORMALIZED_POSITION_NAMES[p] for p in VALID_TEAM_POSITIONS],
        "participant_numeric_features": list(PARTICIPANT_NUMERIC_FEATURES),
        "participant_share_features": list(PARTICIPANT_SHARE_FEATURES),
        "participant_jungle_only_features": list(PARTICIPANT_JUNGLE_ONLY_FEATURES),
        "participant_delta_only_features": list(PARTICIPANT_DELTA_ONLY_FEATURES),
        "challenge_position_features": [spec["name"] for spec in CHALLENGE_POSITION_FEATURES],
        "challenge_jungle_features": [spec["name"] for spec in CHALLENGE_JUNGLE_FEATURES],
        "challenge_delta_only_features": [spec["name"] for spec in CHALLENGE_DELTA_ONLY_FEATURES],
        "team_context_features": list(TEAM_CONTEXT_COLUMNS),
        "objectives_dto_features": list(OBJECTIVES_DTO_COLUMNS),
        "social_context_features": list(SOCIAL_CONTEXT_COLUMNS),
        "sample_random": bool(args.sample_random),
        "sample_seed": int(args.sample_seed),
        "position_rule": {"uses_teamPosition_only": True, "fallback_to_individualPosition": False, "drop_if_teamPosition_missing": True},
        "winner_rule": {"source": "participants[].win", "requires_team_ids": [100, 200], "requires_within_team_consistency": True, "requires_teams_to_differ": True},
        "secondary_time_rule": {
            "server_source": "match_id prefix before first underscore",
            "timestamp_source": "gameStartTimestamp",
            "timezone_mapping": dict(SERVER_TIMEZONE_MAP),
            "fallback_order": [
                "exact platform mapping",
                "source_folder heuristic",
                "platform family heuristic",
                "UTC",
            ],
            "fallback_timezone": "UTC",
        },
        "folder_kept_counts": dict(sorted(folder_kept.items())),
    }
    save_json(output_dir / "dataset_summary.json", summary)
    text_lines = [
        f"input_root={input_root}",
        f"output_dir={output_dir}",
        f"db_path={db_path}",
        f"discovered_files={len(files)}",
        f"loaded_payloads={loaded}",
        f"kept_matches={kept}",
        f"duplicate_match_ids_skipped={duplicate_match_ids}",
        f"dropped_matches={sum(drop_reasons.values())}",
        f"primary_model_feature_count={primary_model_feature_count}",
        f"secondary_analysis_feature_count={secondary_analysis_feature_count}",
        f"non_metadata_column_count={non_metadata_column_count}",
        f"total_column_count={len(schema_feature_columns())}",
    ]
    for reason, count in sorted(drop_reasons.items()):
        text_lines.append(f"drop_reason[{reason}]={count}")
    (output_dir / "dataset_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(f"Built match_table_v1 at: {db_path}")
    print(f"kept_matches={kept}")
    print(f"duplicate_match_ids_skipped={duplicate_match_ids}")
    print(f"dropped_matches={sum(drop_reasons.values())}")
    print(f"primary_model_feature_count={primary_model_feature_count}")
    print(f"secondary_analysis_feature_count={secondary_analysis_feature_count}")
    print(f"non_metadata_column_count={non_metadata_column_count}")
    print(f"total_column_count={len(schema_feature_columns())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

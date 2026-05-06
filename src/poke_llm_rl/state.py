from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PARTY_SIZE_ADDRESS = 0xD163
X_POS_ADDRESS = 0xD362
Y_POS_ADDRESS = 0xD361
MAP_N_ADDRESS = 0xD35E
BADGE_COUNT_ADDRESS = 0xD356
LEVELS_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDRESSES = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
EVENT_FLAGS_START_ADDRESS = 0xD747
EVENT_FLAGS_END_ADDRESS = 0xD886
MUSEUM_TICKET_ADDRESS = 0xD754
MUSEUM_TICKET_BIT = 0
DEFAULT_MAP_DATA_PATH = Path("assets/map_data.json")


@dataclass(slots=True)
class EmulatorState:
    frame: np.ndarray
    map_id: int
    map_name: str
    x: int
    y: int
    badges: int
    party_size: int
    party_levels: list[int]
    hp_fraction: float
    event_flag_count: int
    event_flags: list[int]

    @property
    def unique_tile_key(self) -> tuple[int, int, int]:
        return (self.map_id, self.x, self.y)


def popcount(value: int) -> int:
    return int(value).bit_count()


def read_uint16(memory_reader, start: int) -> int:
    return 256 * int(memory_reader(start)) + int(memory_reader(start + 1))


def read_event_bits(memory_reader) -> list[int]:
    bits: list[int] = []
    for address in range(EVENT_FLAGS_START_ADDRESS, EVENT_FLAGS_END_ADDRESS):
        byte = int(memory_reader(address))
        bits.extend(int(bit) for bit in f"{byte:08b}")
    return bits


def count_event_flags(memory_reader) -> int:
    total = 0
    for address in range(EVENT_FLAGS_START_ADDRESS, EVENT_FLAGS_END_ADDRESS):
        total += popcount(int(memory_reader(address)))
    museum_ticket = int(memory_reader(MUSEUM_TICKET_ADDRESS))
    total -= (museum_ticket >> MUSEUM_TICKET_BIT) & 1
    return max(total, 0)


def hp_fraction(memory_reader) -> float:
    current = sum(read_uint16(memory_reader, address) for address in HP_ADDRESSES)
    maximum = sum(read_uint16(memory_reader, address) for address in MAX_HP_ADDRESSES)
    return current / max(maximum, 1)

@lru_cache(maxsize=8)
def load_map_names(map_data_path: str | Path = DEFAULT_MAP_DATA_PATH) -> dict[int, str]:
    path = Path(map_data_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    names: dict[int, str] = {}
    for region in raw.get("regions", []):
        try:
            region_id = int(region["id"])
        except (KeyError, TypeError, ValueError):
            continue
        region_name = region.get("name")
        if isinstance(region_name, str) and region_name.strip():
            names[region_id] = region_name.strip()
    return names


def map_name(map_id: int, map_data_path: str | Path = DEFAULT_MAP_DATA_PATH) -> str:
    return load_map_names(map_data_path).get(map_id, f"Unknown Map {map_id}")


def extract_emulator_state(frame: np.ndarray, memory_reader, map_data_path: str | Path = DEFAULT_MAP_DATA_PATH) -> EmulatorState:
    current_map_id = int(memory_reader(MAP_N_ADDRESS))
    party_levels = [int(memory_reader(address)) for address in LEVELS_ADDRESSES]
    event_bits = read_event_bits(memory_reader)
    return EmulatorState(
        frame=frame,
        map_id=current_map_id,
        map_name=map_name(current_map_id, map_data_path),
        x=int(memory_reader(X_POS_ADDRESS)),
        y=int(memory_reader(Y_POS_ADDRESS)),
        badges=popcount(int(memory_reader(BADGE_COUNT_ADDRESS))),
        party_size=int(memory_reader(PARTY_SIZE_ADDRESS)),
        party_levels=party_levels,
        hp_fraction=hp_fraction(memory_reader),
        event_flag_count=count_event_flags(memory_reader),
        event_flags=event_bits,
    )

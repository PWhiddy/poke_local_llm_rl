from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


EXPECTED_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"


def sha1sum(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that the Pokemon Red ROM matches the expected SHA1.")
    parser.add_argument("rom_path", nargs="?", default="PokemonRed.gb")
    args = parser.parse_args()
    rom_path = Path(args.rom_path)
    actual = sha1sum(rom_path)
    print(f"path={rom_path}")
    print(f"sha1={actual}")
    print(f"matches_expected={actual == EXPECTED_SHA1}")


if __name__ == "__main__":
    main()

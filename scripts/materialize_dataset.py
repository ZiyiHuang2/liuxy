from __future__ import annotations

import base64
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "voc_dataset_1+2_vs_3.mat"
    parts = sorted(output.parent.glob(output.name + ".b64.part*"))
    if not parts:
        raise FileNotFoundError("No encoded dataset parts found")
    encoded = "".join(part.read_text(encoding="ascii").strip() for part in parts)
    output.write_bytes(base64.b64decode(encoded, validate=True))
    print(output)


if __name__ == "__main__":
    main()

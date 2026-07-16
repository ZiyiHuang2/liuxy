from __future__ import annotations

import base64
import hashlib
from pathlib import Path

EXPECTED_SHA256 = "5abfb996395fc9814cddb266cbde93efab7993dc551450507312469ab0ef2635"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "voc_dataset_1+2_vs_3.mat"

    if not output.exists():
        parts = sorted(output.parent.glob(output.name + ".b64.part*"))
        if not parts:
            raise FileNotFoundError(f"Dataset not found: {output}")
        encoded = "".join(
            part.read_text(encoding="ascii").strip() for part in parts
        )
        output.write_bytes(base64.b64decode(encoded, validate=True))

    actual = _sha256(output)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"Dataset hash mismatch: expected {EXPECTED_SHA256}, got {actual}"
        )

    print(f"Dataset verified: {output}")
    print(f"SHA-256: {actual}")


if __name__ == "__main__":
    main()

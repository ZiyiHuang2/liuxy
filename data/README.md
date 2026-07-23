# Dataset

The repository directly includes the complete original file:

```text
voc_dataset_1+2_vs_3.mat
```

## Schema

| Key | Shape | Meaning |
|---|---:|---|
| `X` | 159 × 445 | Continuous VOC feature matrix |
| `y` | 159 × 1 | Binary target |
| `feat_names` | 445 | VOC feature names |

Label mapping:

- `0`: original classes 1 and 2 combined, 106 samples
- `1`: original class 3, 53 samples

## Integrity

File size:

```text
341,128 bytes
```

SHA-256:

```text
5abfb996395fc9814cddb266cbde93efab7993dc551450507312469ab0ef2635
```

Git blob SHA:

```text
d72ecbd7e4770375b76a37224a919c517e0befc3
```

The blob SHA matches the source file in `liu2690/my-project`.

Run the integrity check with:

```bash
python scripts/materialize_dataset.py
```

## Provenance and retained Unknown payload

The verified MAT binary is committed directly to the repository, so normal training and evaluation require no network access. Its SHA-256 is fixed above.

The auxiliary directory `results/unknown_voc_extract/` contains a chunked, base64-encoded NPZ payload reconstructed from the original raw VOC table. `manifest.json` lists the ordered parts and fixes the decoded payload SHA-256 to:

```text
2040c38df075e71b3a588bc94a588dc5d3d3e0c0bb7158808c2bea7da8dabff8
```

The decoded payload contains:

| Key | Shape | Meaning |
|---|---:|---|
| `combined_X` | 159 × 780 | known and Unknown columns filtered together |
| `unknown_X` | 159 × 335 | Unknown columns filtered independently |
| `y` | 159 | labels verified against the primary MAT |

`experiments/run_unknown_voc_comparison.py` reconstructs and verifies this payload automatically. It also forms the 780-feature appended matrix by concatenating the frozen 445-feature known matrix and the 335-feature Unknown-only matrix.

## Important limitation

The file does not contain subject identifiers, sampling batch, device, date or other grouping metadata. Cross-validation therefore operates at the row level. It cannot prove subject-level independence or rule out hidden batch effects.

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

## Provenance

`.github/workflows/import-dataset.yml` downloaded the file from the original public project repository, checked the fixed SHA-256 and committed the verified binary to this repository. The experiment code does not require network access after cloning this repository.

## Important limitation

The file does not contain subject identifiers, sampling batch, device, date or other grouping metadata. Cross-validation therefore operates at the row level. It cannot prove subject-level independence or rule out hidden batch effects.

# Dataset

This repository includes the full `voc_dataset_1+2_vs_3.mat` payload as numbered base64 parts so the experiments are self-contained. The Python loader decodes the parts transparently when the raw MAT file is absent.

## Schema

| Key | Shape | Meaning |
|---|---:|---|
| `X` | 159 × 445 | Continuous VOC feature matrix |
| `y` | 159 × 1 | Binary target |
| `feat_names` | 445 | VOC feature names |

Label mapping:

- `0`: original classes 1 and 2 combined, 106 samples
- `1`: original class 3, 53 samples

SHA-256:

```text
5abfb996395fc9814cddb266cbde93efab7993dc551450507312469ab0ef2635
```

## Storage format

The 341,128-byte MAT payload is stored losslessly as:

```text
voc_dataset_1+2_vs_3.mat.b64.part01
...
voc_dataset_1+2_vs_3.mat.b64.part10
```

The package loads these parts automatically. To recreate the physical MAT file:

```bash
python scripts/materialize_dataset.py
```

## Important limitation

The file does not contain subject identifiers, sampling batch, device, date, or other grouping metadata. Cross-validation therefore operates at the row level. It cannot prove subject-level independence or rule out hidden batch effects.

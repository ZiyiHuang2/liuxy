from pathlib import Path

import pandas as pd

from voc_easyensemble.data import load_voc_mat


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = load_voc_mat(root / "data" / "voc_dataset_1+2_vs_3.mat")
    output = root / "data" / "new_samples_template.csv"
    pd.DataFrame(columns=dataset.feature_names).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()

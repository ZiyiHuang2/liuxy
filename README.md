# VOC EasyEnsemble Classification

这是小样本 VOC 二分类项目的最终稳健实现。

当前主模型：

> **SNV + ANOVA Top-125 + EasyEnsemble-50 + fixed threshold 0.5**

## Motivation

原始版本尝试使用 MLP/CNN 直接学习 VOC 特征，但该任务具有明显的小样本、高维、不平衡特点：

- 159 个样本
- 445 个 VOC 特征
- 正负样本比例约 1:2

因此最终方案转向稳定的小样本表格学习流程。

## Pipeline

```
VOC matrix
  -> sample-wise SNV normalization
  -> feature selection inside training folds (ANOVA Top-125)
  -> 50 balanced training subsets
  -> AdaBoost decision stumps
  -> probability averaging
  -> threshold = 0.5
```

## Reliable Results

在 16 个未参与模型选择的重复外层验证随机种子上：

| Metric | Result |
|---|---:|
| F1 | 0.7615 ± 0.0357 |
| ROC-AUC | 0.8971 |
| PR-AUC | 0.8268 |

## Project Structure

```
.
├── src/voc_easyensemble
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── configs
├── docs
├── results
└── tests
```

## Usage

Install:

```bash
pip install -r requirements.txt
```

Train:

```bash
python -m voc_easyensemble.train --config configs/default.json
```

Evaluate:

```bash
python -m voc_easyensemble.evaluate
```

## Reproducibility

All feature selection and preprocessing operations must be fitted only on training folds. Test folds are never used for feature selection or model selection.

See:

- `docs/METHOD.md`
- `docs/RESULTS.md`
- `results/reliable_round5/`

# 第六轮自动探索报告

## 结论

本轮最高重复确认候选为：

> **SNV + 三种折内特征子空间 EasyEnsemble-50 + 等权概率平均 + 固定阈值 0.5**

三个分支均只使用外层训练折计算的 ANOVA 分数：

1. 固定 ANOVA Top-125；
2. 每个平衡子模型按 ANOVA 分数，从全部 445 个 VOC 中无放回抽取 150 个；
3. 每个平衡子模型从 ANOVA Top-250 中均匀抽取 125 个。

每个分支包含 50 个 AdaBoost 树桩模型，三路正类概率等权平均。

## 32个全新随机种子确认

| 方法 | F1均值 | F1标准差 | 最差F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **三路特征多样性集成** | **0.7714** | 0.0286 | **0.7037** | **0.9018** | **0.8319** |
| 固定 ANOVA Top-125 Easy-50 | 0.7658 | 0.0308 | 0.6852 | 0.8993 | 0.8297 |

配对结果：平均 F1 提升 `+0.00556`，17胜、3平、12负；bootstrap 95% CI 约为 `[-0.00002, +0.01126]`，Wilcoxon `p=0.0938`。

因此它是当前最高增强候选，但不能声称已经统计上确定优于固定 Top-125。

## canonical seed=42

| 方法 | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| 三路特征多样性集成 | **0.7593** | 0.8884 | 0.8264 |
| 固定 Top-125 | 0.7321 | 0.8873 | 0.8205 |

单个 seed 只作为固定划分核对，正式结论以32种子配对结果为准。

## 阈值确认

0.495 阈值在16个新种子上平均 F1 为0.7520，低于0.5阈值的0.7565，因此最终继续使用固定0.5。

## 被否定的方向

- PLS、Elastic Net/L1 Logistic、RBF-SVM、Shrinkage LDA、正则QDA；
- VOC对数比、成对差分和相关模块聚合；
- 内层OOF自适应权重和阈值；
- 高权重化学功能组融合；
- 阈值0.495。

## 提升来源

原EasyEnsemble只在样本维度随机化负类子集。本轮同时在特征维度构造不同的高证据VOC子空间，以降低对单一Top-125 panel的依赖。

## 复现文件

- `src/voc_easyensemble/feature_diverse.py`；
- `configs/feature_diverse.json`；
- `results/round6/confirm32_rows_*.csv`；
- `results/round6/confirm32_summary.csv`；
- `results/round6/confirm32_paired.csv`；
- `results/round6/threshold_confirm16_summary.csv`；
- `results/round6/experiment_manifest.json`。

```bash
voc-easy evaluate \
  --config configs/feature_diverse.json \
  --seeds 85001:85016,86001:86016 \
  --output outputs/feature_diverse_32seed
```

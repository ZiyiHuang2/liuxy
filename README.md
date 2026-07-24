# VOC EasyEnsemble 小样本分类

本仓库用于高维、小样本 VOC 二分类实验，重点验证 EasyEnsemble、特征子空间集成，以及是否应保留名称为 `Unknown` 的 VOC 特征。

## 1. 数据与任务

- 样本数：159；
- 类别分布：106 vs. 53；
- 删除精确名称为 `Unknown` 的列后，known 特征数为 445；
- 单独过滤后可保留 335 个 Unknown 特征；
- known 与 Unknown 拼接后共有 780 个特征；
- 原始 MAT 数据和 Unknown payload 均通过 SHA-256 校验。

> 重要限制：数据中没有受试者编号、采集批次、设备或采样日期，因此当前交叉验证只能保证行级划分，无法证明受试者级独立，也无法完全排除隐藏批次效应。

## 2. 当前模型与正式结论

当前保留两条模型线：

| 模型 | 主要流程 | 定位 |
|---|---|---|
| 稳定基线 | SNV → ANOVA Top-125 → EasyEnsemble-50 → 阈值 0.5 | 简单、稳定、便于复现 |
| 特征多样性模型 | SNV → 三个 VOC 子空间分支 → 3 × EasyEnsemble-50 → 概率平均 | 当前数值最好的增强候选 |

特征多样性模型在此前 32 个随机种子上的平均 F1 为 `0.7714 ± 0.0286`，高于固定 Top-125 基线的 `0.7658 ± 0.0308`。但配对提升较小，Wilcoxon `p=0.0938`，因此应表述为“当前最佳增强候选”，而不是已经统计显著优于基线。

### 关于 Unknown 的最终结论

重新规划并跑满完整实验后，**正式模型继续删除 Unknown**。

最终 32 个全新随机种子的结果如下：

| 方法 | 平均 F1 | 标准差 | 最低 F1 |
|---|---:|---:|---:|
| **known-only 基线** | **0.774223** | **0.024776** | **0.718447** |
| 全部 Unknown，5% 概率融合 | 0.772497 | 0.027225 | 0.718447 |
| 稳定 Unknown Top-150，10% 融合 | 0.771545 | 0.029324 | 0.715596 |
| 稳定 Unknown Top-150，20% 融合 | 0.770610 | 0.034233 | 0.712871 |
| 全部 Unknown，20% 概率融合 | 0.770142 | 0.034528 | 0.693069 |

最好的 Unknown 方案是 5% 概率融合，但相对 known-only：

- 平均配对差值：`-0.001727`；
- bootstrap 95% CI：`[-0.004899, 0.001411]`；
- 9 胜、9 平、14 负；
- Wilcoxon `p=0.3304`。

因此，没有证据支持用 Unknown 方案替换当前 known-only 模型。完整报告见 [`results/unknown_voc_replanned/REPORT.md`](results/unknown_voc_replanned/REPORT.md)。

## 3. 特征多样性模型

增强模型在每个训练折内训练三个分支：

1. **固定强特征分支**：ANOVA Top-125；
2. **加权探索分支**：每个平衡子模型根据当前训练折的 ANOVA 分数，从全部 445 个 VOC 中抽取 150 个特征；
3. **Top-pool 多样性分支**：每个平衡子模型从当前训练折的 ANOVA Top-250 中抽取 125 个特征。

每个分支包含 50 个 AdaBoost 子模型，每个 AdaBoost 使用 50 棵深度为 1 的树。每个子模型使用全部正样本，并随机抽取等量负样本；三个分支的预测概率取平均，分类阈值固定为 0.5。

## 4. Unknown 完整实验协议

这次实验不再只从轻量阶段选两个“冠军”，而是采用三阶段独立随机种子协议：

| 阶段 | 随机种子 | 模型预算 | 候选传递 |
|---|---:|---:|---:|
| Screen | 12 | 每分支 12 个子模型，25 棵树 | 22 个候选中选 6 个 |
| Verify | 16 | 每分支 25 个子模型，35 棵树 | 6 个中选 4 个 |
| Final | 32 | 每分支 50 个子模型，50 棵树 | 报告前 3 名并与基线配对检验 |

候选包括：

- combined、appended、unknown-only 直接建模；
- 在每个外层训练折内稳定筛选 Unknown Top-25、50、100、150；
- stable Unknown 与 known 直接拼接；
- known 与 Unknown 按 5%、10%、20% 概率融合。

Screen、Verify 和 Final 使用互不重叠的种子，Unknown 特征筛选只使用当前训练折，避免数据泄漏。候选只有同时满足以下条件才允许替换 known-only：

- 配对平均提升为正；
- bootstrap 95% CI 下界大于 0；
- Wilcoxon `p<0.05`；
- 胜场多于负场；
- 最低 F1 不比基线低超过 0.005。

本次所有 Unknown 候选均未达到替换标准。

## 5. 安装

推荐 Python 3.10 或更高版本：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

也可以直接安装运行依赖：

```bash
python -m pip install -r requirements.txt
```

## 6. 常用命令

### 运行测试

```bash
pytest
```

### 快速检查 Unknown 三阶段流程

```bash
make unknown-replan-quick
```

### 运行完整 Unknown 实验

```bash
make unknown-replan-full
```

等价命令：

```bash
python experiments/run_unknown_voc_replanned.py \
  --config configs/unknown_voc_replanned.json \
  --output results/unknown_voc_replanned
```

脚本按种子保存 checkpoint；中断后使用相同配置和输出目录重新运行，只会补算缺失种子。

### 复现特征多样性 32-seed 实验

```bash
voc-easy evaluate \
  --config configs/feature_diverse.json \
  --seeds 85001:85016,86001:86016 \
  --output outputs/feature_diverse_32seed
```

### 训练与预测

```bash
voc-easy train \
  --config configs/feature_diverse.json \
  --model artifacts/voc_feature_diverse.joblib

voc-easy predict \
  --model artifacts/voc_feature_diverse.joblib \
  --input new_samples.csv \
  --output outputs/predictions.csv
```

## 7. 仓库结构

```text
.
├── configs/
│   ├── default.json
│   ├── feature_diverse.json
│   └── unknown_voc_replanned.json
├── data/
│   └── voc_dataset_1+2_vs_3.mat
├── experiments/
│   ├── run_unknown_voc_comparison.py
│   └── run_unknown_voc_replanned.py
├── results/
│   ├── round6/
│   ├── unknown_voc/
│   ├── unknown_voc_extract/
│   └── unknown_voc_replanned/
├── src/voc_easyensemble/
└── tests/
```

## 8. 结果文件说明

`results/unknown_voc_replanned/` 中保存：

- `REPORT.md`：中文完整报告；
- `manifest.json`：冻结配置、随机种子、候选和数据哈希；
- `*_rows.csv`：每个随机种子的原始指标；
- `*_summary.csv`：各阶段汇总；
- `*_paired.csv`：相对 known-only 的配对检验；
- `*_selected.json`：各阶段进入下一阶段的候选。

## 9. 相关文档

- [方法说明](docs/METHOD.md)
- [历史结果](docs/RESULTS.md)
- [实验历史](docs/EXPERIMENT_HISTORY.md)
- [Round-6 报告](docs/ROUND6_EXPLORATION.md)
- [第一次 Unknown 实验报告](results/unknown_voc/REPORT.md)
- [Unknown 重新规划协议](docs/UNKNOWN_VOC_REPLAN.md)
- [Unknown 完整最终报告](results/unknown_voc_replanned/REPORT.md)
- [数据说明](data/README.md)

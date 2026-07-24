# Unknown VOC 完整实验结果

本目录保存沙盒中完成的三阶段完整实验：

- Screen：12 个随机种子；
- Verify：16 个全新随机种子；
- Final：32 个全新随机种子；
- 五折外层交叉验证；
- Final 每个分支 50 个子模型、每个 AdaBoost 50 棵树；
- Unknown 稳定性筛选仅在当前外层训练折内进行。

正式结论：所有 Unknown 候选均未达到替换 known-only 基线的预设标准，正式模型继续删除 Unknown。

优先阅读：

- `REPORT.md`：中文完整报告；
- `final_summary.csv`：最终汇总；
- `final_paired.csv`：最终配对检验；
- `manifest.json`：冻结配置、随机种子、数据哈希和候选传递。

# Unknown VOC 完整实验结果

本目录保存沙盒中完成的三阶段完整实验：

- Screen：12 个随机种子；
- Verify：16 个全新随机种子；
- Final：32 个全新随机种子；
- 五折外层交叉验证；
- Final 每个分支 50 个子模型、每个 AdaBoost 50 棵树；
- Unknown 稳定性筛选仅在当前外层训练折内进行。

正式结论：所有 Unknown 候选均未达到替换 known-only 基线的预设标准，正式模型继续删除 Unknown。

## 文件说明

- `REPORT.md`：中文完整报告；
- `manifest.json`：冻结配置、随机种子、数据哈希和候选传递；
- `screen_summary.csv`、`verify_summary.csv`、`final_summary.csv`：三阶段汇总；
- `screen_paired.csv`、`verify_paired.csv`、`final_paired.csv`：相对 known-only 的配对检验；
- `verify_rows.csv`、`final_rows.csv`：Verify 和 Final 的逐种子原始指标；
- `screen_rows.csv.gz.b64`：Screen 逐种子原始指标的 gzip + Base64 文本；
- `*_selected.json`：每个阶段冻结进入下一阶段的候选。

解码 Screen 原始记录：

```bash
base64 -d results/unknown_voc_replanned/screen_rows.csv.gz.b64 \
  | gzip -d \
  > results/unknown_voc_replanned/screen_rows.csv
```

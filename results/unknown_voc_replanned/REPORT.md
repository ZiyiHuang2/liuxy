# Unknown VOC 重新规划实验

## 最终判定

正式模型继续删除 Unknown。

该实验不再用一次 `.head(2)` 决定全部结论，而是采用分阶段冠军池：

- 筛选阶段进入复核（6 个）：`known_unknown_stable_k150_w10`, `known_unknown_all_w20`, `known_unknown_stable_k150_w20`, `known_unknown_all_w05`, `appended_stable_k50`, `appended_all_diverse`；
- 复核阶段进入最终确认（4 个）：`known_unknown_stable_k150_w20`, `known_unknown_stable_k150_w10`, `known_unknown_all_w05`, `known_unknown_all_w20`；
- 最终阶段保留报告冠军（3 个）：`known_unknown_all_w05`, `known_unknown_stable_k150_w10`, `known_unknown_stable_k150_w20`。

所有 Unknown 特征选择均只使用当前外层训练折；筛选、复核和最终确认使用互不重叠的随机种子。

## 数据审计

- known：[159, 445]；
- combined：[159, 780]；
- unknown-only：[159, 335]；
- appended-all：[159, 780]。

## screen 阶段

| method                        | family        |   n_seeds |   f1_mean |   f1_std |   f1_min |   f1_max |   roc_auc_mean |   pr_auc_mean |
|:------------------------------|:--------------|----------:|----------:|---------:|---------:|---------:|---------------:|--------------:|
| known_unknown_stable_k150_w10 | stable_fusion |        12 |  0.761081 | 0.037905 | 0.673077 | 0.810811 |       0.891480 |      0.819081 |
| known_unknown_all_w20         | all_fusion    |        12 |  0.760053 | 0.038630 | 0.666667 | 0.810811 |       0.889670 |      0.815145 |
| known_unknown_stable_k150_w20 | stable_fusion |        12 |  0.759366 | 0.031818 | 0.692308 | 0.803738 |       0.889922 |      0.817384 |
| known_unknown_stable_k100_w10 | stable_fusion |        12 |  0.758469 | 0.037210 | 0.673077 | 0.810811 |       0.890797 |      0.817084 |
| known_unknown_stable_k50_w10  | stable_fusion |        12 |  0.758127 | 0.038635 | 0.666667 | 0.821429 |       0.888558 |      0.813976 |
| known_unknown_stable_k50_w20  | stable_fusion |        12 |  0.756926 | 0.035018 | 0.679245 | 0.810811 |       0.881586 |      0.805252 |
| known_unknown_stable_k100_w20 | stable_fusion |        12 |  0.755510 | 0.033513 | 0.673077 | 0.803571 |       0.888068 |      0.814449 |
| known_unknown_all_w05         | all_fusion    |        12 |  0.755343 | 0.036473 | 0.660377 | 0.803571 |       0.891851 |      0.817130 |
| known_unknown_all_w10         | all_fusion    |        12 |  0.755258 | 0.036854 | 0.660377 | 0.803571 |       0.891925 |      0.817670 |
| known_unknown_stable_k50_w05  | stable_fusion |        12 |  0.754188 | 0.038402 | 0.666667 | 0.814159 |       0.890723 |      0.816712 |
| known_unknown_stable_k150_w05 | stable_fusion |        12 |  0.754165 | 0.039190 | 0.660377 | 0.803571 |       0.892058 |      0.818158 |
| appended_stable_k50           | stable_append |        12 |  0.753734 | 0.043927 | 0.678261 | 0.818182 |       0.881349 |      0.815667 |
| known_unknown_stable_k100_w05 | stable_fusion |        12 |  0.753547 | 0.036380 | 0.660377 | 0.803571 |       0.891836 |      0.817644 |
| known_unknown_stable_k25_w05  | stable_fusion |        12 |  0.752847 | 0.035214 | 0.654206 | 0.792793 |       0.889596 |      0.814423 |
| known_diverse                 | baseline      |        12 |  0.751846 | 0.033167 | 0.660377 | 0.785714 |       0.891139 |      0.816146 |
| known_unknown_stable_k25_w10  | stable_fusion |        12 |  0.750962 | 0.038074 | 0.660377 | 0.792793 |       0.886362 |      0.811258 |
| appended_all_diverse          | direct        |        12 |  0.750200 | 0.043132 | 0.647059 | 0.807339 |       0.872820 |      0.811096 |
| appended_stable_k25           | stable_append |        12 |  0.746479 | 0.046579 | 0.684685 | 0.810811 |       0.883321 |      0.813684 |
| appended_stable_k150          | stable_append |        12 |  0.742445 | 0.030705 | 0.685714 | 0.777778 |       0.872923 |      0.811128 |
| appended_stable_k100          | stable_append |        12 |  0.734815 | 0.029315 | 0.673077 | 0.766355 |       0.870802 |      0.806253 |
| known_unknown_stable_k25_w20  | stable_fusion |        12 |  0.733937 | 0.039186 | 0.647059 | 0.774775 |       0.876780 |      0.800204 |
| combined_diverse              | direct        |        12 |  0.717579 | 0.031618 | 0.666667 | 0.774775 |       0.863133 |      0.796659 |
| unknown_all_diverse           | direct        |        12 |  0.603955 | 0.036605 | 0.533333 | 0.672269 |       0.767474 |      0.661761 |

### 相对 known 基线的配对结果

| candidate                     | family        | baseline      |   n |   mean_delta_f1 |   median_delta_f1 |   ci95_low |   ci95_high |   wins |   ties |   losses |   wilcoxon_statistic |   wilcoxon_p |
|:------------------------------|:--------------|:--------------|----:|----------------:|------------------:|-----------:|------------:|-------:|-------:|---------:|---------------------:|-------------:|
| combined_diverse              | direct        | known_diverse |  12 |       -0.034266 |         -0.037578 |  -0.047169 |   -0.020955 |      1 |      0 |       11 |             2.000000 |     0.001465 |
| appended_all_diverse          | direct        | known_diverse |  12 |       -0.001646 |         -0.002273 |  -0.016712 |    0.013904 |      5 |      1 |        6 |            32.500000 |     0.983398 |
| unknown_all_diverse           | direct        | known_diverse |  12 |       -0.147891 |         -0.150959 |  -0.168213 |   -0.126844 |      0 |      0 |       12 |             0.000000 |     0.000488 |
| appended_stable_k25           | stable_append | known_diverse |  12 |       -0.005367 |         -0.001560 |  -0.022851 |    0.012355 |      6 |      0 |        6 |            33.000000 |     0.677246 |
| appended_stable_k50           | stable_append | known_diverse |  12 |        0.001889 |          0.003548 |  -0.015916 |    0.019234 |      7 |      0 |        5 |            33.000000 |     0.677246 |
| appended_stable_k100          | stable_append | known_diverse |  12 |       -0.017030 |         -0.013579 |  -0.028135 |   -0.006413 |      3 |      0 |        9 |             9.000000 |     0.015625 |
| appended_stable_k150          | stable_append | known_diverse |  12 |       -0.009400 |         -0.012242 |  -0.022814 |    0.003244 |      5 |      0 |        7 |            23.000000 |     0.233398 |
| known_unknown_stable_k25_w05  | stable_fusion | known_diverse |  12 |        0.001002 |          0.000000 |  -0.003171 |    0.005891 |      4 |      3 |        5 |            21.000000 |     0.910156 |
| known_unknown_stable_k25_w10  | stable_fusion | known_diverse |  12 |       -0.000884 |          0.000000 |  -0.010210 |    0.008223 |      5 |      2 |        5 |            27.000000 |     0.976562 |
| known_unknown_stable_k25_w20  | stable_fusion | known_diverse |  12 |       -0.017909 |         -0.022292 |  -0.031630 |   -0.003867 |      4 |      0 |        8 |            12.000000 |     0.034180 |
| known_unknown_stable_k50_w05  | stable_fusion | known_diverse |  12 |        0.002342 |          0.003145 |  -0.006573 |    0.011105 |      6 |      1 |        5 |            29.000000 |     0.764648 |
| known_unknown_stable_k50_w10  | stable_fusion | known_diverse |  12 |        0.006281 |          0.001147 |  -0.002563 |    0.015940 |      6 |      3 |        3 |            14.000000 |     0.359375 |
| known_unknown_stable_k50_w20  | stable_fusion | known_diverse |  12 |        0.005080 |          0.003663 |  -0.006212 |    0.016822 |      6 |      2 |        4 |            20.000000 |     0.492188 |
| known_unknown_stable_k100_w05 | stable_fusion | known_diverse |  12 |        0.001702 |          0.000000 |  -0.003571 |    0.007084 |      5 |      3 |        4 |            19.000000 |     0.734375 |
| known_unknown_stable_k100_w10 | stable_fusion | known_diverse |  12 |        0.006623 |          0.010013 |  -0.001117 |    0.013984 |      8 |      1 |        3 |            17.000000 |     0.174805 |
| known_unknown_stable_k100_w20 | stable_fusion | known_diverse |  12 |        0.003664 |          0.004414 |  -0.001790 |    0.008796 |      7 |      2 |        3 |            14.000000 |     0.193359 |
| known_unknown_stable_k150_w05 | stable_fusion | known_diverse |  12 |        0.002320 |          0.000000 |  -0.005014 |    0.008799 |      5 |      5 |        2 |             9.000000 |     0.437500 |
| known_unknown_stable_k150_w10 | stable_fusion | known_diverse |  12 |        0.009235 |          0.009965 |   0.000655 |    0.018074 |      7 |      1 |        4 |            13.000000 |     0.081055 |
| known_unknown_stable_k150_w20 | stable_fusion | known_diverse |  12 |        0.007520 |          0.004290 |  -0.002245 |    0.017400 |      7 |      1 |        4 |            18.000000 |     0.206055 |
| known_unknown_all_w05         | all_fusion    | known_diverse |  12 |        0.003497 |          0.004177 |  -0.001597 |    0.008581 |      7 |      3 |        2 |            13.000000 |     0.300781 |
| known_unknown_all_w10         | all_fusion    | known_diverse |  12 |        0.003412 |          0.005606 |  -0.004197 |    0.010464 |      7 |      2 |        3 |            19.000000 |     0.416016 |
| known_unknown_all_w20         | all_fusion    | known_diverse |  12 |        0.008208 |          0.007354 |  -0.001917 |    0.018751 |      7 |      0 |        5 |            22.000000 |     0.203613 |

## verify 阶段

| method                        | family        |   n_seeds |   f1_mean |   f1_std |   f1_min |   f1_max |   roc_auc_mean |   pr_auc_mean |
|:------------------------------|:--------------|----------:|----------:|---------:|---------:|---------:|---------------:|--------------:|
| known_unknown_stable_k150_w20 | stable_fusion |        16 |  0.770233 | 0.024278 | 0.733945 | 0.821429 |       0.896438 |      0.823111 |
| known_unknown_stable_k150_w10 | stable_fusion |        16 |  0.769368 | 0.021190 | 0.740741 | 0.824561 |       0.898273 |      0.825759 |
| known_unknown_all_w05         | all_fusion    |        16 |  0.767650 | 0.020698 | 0.733945 | 0.817391 |       0.898296 |      0.825798 |
| known_unknown_all_w20         | all_fusion    |        16 |  0.765337 | 0.023115 | 0.725664 | 0.807018 |       0.897172 |      0.823752 |
| known_diverse                 | baseline      |        16 |  0.764336 | 0.025103 | 0.727273 | 0.827586 |       0.897851 |      0.824713 |
| appended_stable_k50           | stable_append |        16 |  0.744577 | 0.024126 | 0.700855 | 0.793103 |       0.884289 |      0.817576 |
| appended_all_diverse          | direct        |        16 |  0.736687 | 0.030532 | 0.678571 | 0.796296 |       0.877370 |      0.812622 |

### 相对 known 基线的配对结果

| candidate                     | family        | baseline      |   n |   mean_delta_f1 |   median_delta_f1 |   ci95_low |   ci95_high |   wins |   ties |   losses |   wilcoxon_statistic |   wilcoxon_p |
|:------------------------------|:--------------|:--------------|----:|----------------:|------------------:|-----------:|------------:|-------:|-------:|---------:|---------------------:|-------------:|
| known_unknown_stable_k150_w10 | stable_fusion | known_diverse |  16 |        0.005032 |          0.006783 |  -0.001629 |    0.010865 |     10 |      2 |        4 |            24.500000 |     0.078718 |
| known_unknown_all_w20         | all_fusion    | known_diverse |  16 |        0.001001 |          0.005007 |  -0.007303 |    0.008466 |      9 |      1 |        6 |            54.000000 |     0.733271 |
| known_unknown_stable_k150_w20 | stable_fusion | known_diverse |  16 |        0.005897 |          0.003252 |  -0.004102 |    0.015987 |     10 |      1 |        5 |            41.000000 |     0.280531 |
| known_unknown_all_w05         | all_fusion    | known_diverse |  16 |        0.003314 |          0.003260 |  -0.002206 |    0.008571 |      8 |      5 |        3 |            20.000000 |     0.247746 |
| appended_stable_k50           | stable_append | known_diverse |  16 |       -0.019759 |         -0.023736 |  -0.030011 |   -0.009239 |      3 |      0 |       13 |            15.000000 |     0.004181 |
| appended_all_diverse          | direct        | known_diverse |  16 |       -0.027649 |         -0.038273 |  -0.044455 |   -0.009557 |      4 |      0 |       12 |            20.000000 |     0.010986 |

## final 阶段

| method                        | family        |   n_seeds |   f1_mean |   f1_std |   f1_min |   f1_max |   roc_auc_mean |   pr_auc_mean |
|:------------------------------|:--------------|----------:|----------:|---------:|---------:|---------:|---------------:|--------------:|
| known_diverse                 | baseline      |        32 |  0.774223 | 0.024776 | 0.718447 | 0.825688 |       0.901227 |      0.833238 |
| known_unknown_all_w05         | all_fusion    |        32 |  0.772497 | 0.027225 | 0.718447 | 0.836364 |       0.901661 |      0.834605 |
| known_unknown_stable_k150_w10 | stable_fusion |        32 |  0.771545 | 0.029324 | 0.715596 | 0.836364 |       0.901616 |      0.835306 |
| known_unknown_stable_k150_w20 | stable_fusion |        32 |  0.770610 | 0.034233 | 0.712871 | 0.844037 |       0.899225 |      0.831623 |
| known_unknown_all_w20         | all_fusion    |        32 |  0.770142 | 0.034528 | 0.693069 | 0.833333 |       0.900604 |      0.832834 |

### 相对 known 基线的配对结果

| candidate                     | family        | baseline      |   n |   mean_delta_f1 |   median_delta_f1 |   ci95_low |   ci95_high |   wins |   ties |   losses |   wilcoxon_statistic |   wilcoxon_p |
|:------------------------------|:--------------|:--------------|----:|----------------:|------------------:|-----------:|------------:|-------:|-------:|---------:|---------------------:|-------------:|
| known_unknown_stable_k150_w20 | stable_fusion | known_diverse |  32 |       -0.003613 |         -0.005013 |  -0.011648 |    0.004364 |     10 |      2 |       20 |           174.000000 |     0.228880 |
| known_unknown_stable_k150_w10 | stable_fusion | known_diverse |  32 |       -0.002679 |          0.000000 |  -0.008318 |    0.002814 |     12 |      6 |       14 |           137.000000 |     0.328143 |
| known_unknown_all_w05         | all_fusion    | known_diverse |  32 |       -0.001727 |          0.000000 |  -0.004899 |    0.001411 |      9 |      9 |       14 |           106.000000 |     0.330387 |
| known_unknown_all_w20         | all_fusion    | known_diverse |  32 |       -0.004082 |         -0.007572 |  -0.011430 |    0.003342 |     11 |      1 |       20 |           173.000000 |     0.141632 |

## 替换标准

最终候选必须同时满足：配对均值提升为正、bootstrap 95% 区间下界大于 0、Wilcoxon p<0.05、胜场多于负场，且最低 F1 不得比 known 基线低超过 0.005。

# MDQM9/TITO 复现分析报告 / MDQM9/TITO Reproduction Analysis Report

## 1. 摘要 / Abstract

**中文。** 本报告总结了在 MDQM9 test split 上对 TITO 进行论文级推理复现的当前进展。我们已经完成了 `lag=1000 ps`、`nested_samples=1000`、`ode_steps=20` 条件下的全量 1251 个小分子分析，并在 `lag_vamp=5/10/15` 上完成 VAMP-gap 敏感性评估。结果显示，典型分子的 VAMP-gap 接近 0，`lag_vamp=10` 时 signed median 为 `0.0131`，absolute median 为 `0.0970`；但 signed mean 为负，主要由一组强负向 outlier 拉低。

**English.** This report summarizes the current MDQM9/TITO reproduction status. We completed full-test-set analysis for 1251 small molecules using `lag=1000 ps`, `nested_samples=1000`, and `ode_steps=20`, with VAMP-gap sensitivity analyses at `lag_vamp=5/10/15`. The typical molecule has a VAMP-gap close to zero; at `lag_vamp=10`, the signed median is `0.0131` and the absolute median is `0.0970`. The signed mean is negative because of a heavy negative outlier tail.

**关键修正。** 当前 `lag=1000 ps` 结果不应被写成原文 Fig. 2 的严格同口径复现。原文 Fig. 2 小分子参数应按 Supplementary Table S3 对齐为 `lag=57 ps`、`nested samples=640`、`ODE steps=20`、`batch size=32`。原文没有显式报告名为 `lag_vamp` 的参数；`lag_vamp` 是本仓库 `scripts/analyse.py` 中的分析参数。

**Key correction.** The current `lag=1000 ps` results should not be reported as a strictly parameter-matched reproduction of Fig. 2 in the paper. For small molecules, Supplementary Table S3 indicates `lag=57 ps`, `nested samples=640`, `ODE steps=20`, and `batch size=32`. The paper does not explicitly report a parameter named `lag_vamp`; this is a parameter in the repository analysis script.

## 2. 原文口径 / Original Paper Protocol

**中文。** 原文 Fig. 2 标题为 “TITO accurately predicts both thermodynamics and kinetics.” 该图将小分子和四肽分开报告，并聚合评估三类指标：TIC 子空间中的 Jensen-Shannon divergence、VAMP-2 gap、top 10 relative error。对小分子，图中数值为：

| 指标 / Metric | 原文 Fig. 2 小分子结果 / Original Fig. 2 Small-Molecule Result |
|---|---:|
| Jensen-Shannon divergence, mean / median | `0.097 / 0.087` |
| VAMP-2 gap, mean / median | `-0.388 / -0.072` |
| Top-10 relative error, mean / median | `0.554 / 0.192` |

**English.** Fig. 2 in the paper reports small molecules and tetrapeptides separately, using three aggregate metrics: Jensen-Shannon divergence in TIC space, VAMP-2 gap, and top-10 relative error. For small molecules, the reported values are:

| Metric | Original Fig. 2 Small-Molecule Result |
|---|---:|
| Jensen-Shannon divergence, mean / median | `0.097 / 0.087` |
| VAMP-2 gap, mean / median | `-0.388 / -0.072` |
| Top-10 relative error, mean / median | `0.554 / 0.192` |

**采样参数对齐。** 原文 Supplementary Table S3 报告 Fig. 2 的 `Lag = 57 ps/250 ps`，并说明斜杠前后分别对应 small molecules / tetrapeptides。因此 MDQM9 小分子的 Fig. 2 对齐采样 lag 是 `57 ps`，不是 `1000 ps`。

**Sampling protocol alignment.** Supplementary Table S3 reports `Lag = 57 ps/250 ps` for Fig. 2, where the first number corresponds to small molecules and the second to tetrapeptides. Therefore, the Fig. 2 matched MDQM9 small-molecule sampling lag is `57 ps`, not `1000 ps`.

## 3. 当前复现设置 / Current Reproduction Setup

**中文。** 当前已完成分析使用的是长时步扩展设置：

| 参数 / Parameter | 当前值 / Current Value |
|---|---:|
| Dataset split | `mdqm9`, `version_0`, `test` |
| Molecules | `1251/1251` |
| TITO sampling lag | `1000 ps` |
| Nested samples | `1000` |
| ODE steps | `20` |
| TICA lag | `1` |
| VAMP analysis lags | `5`, `10`, `15` |
| Batch size used in sampling outputs | `32` as reflected by output shapes such as `32x1001x2` |

**English.** The completed analysis uses a long-step extension setting:

| Parameter | Current Value |
|---|---:|
| Dataset split | `mdqm9`, `version_0`, `test` |
| Molecules | `1251/1251` |
| TITO sampling lag | `1000 ps` |
| Nested samples | `1000` |
| ODE steps | `20` |
| TICA lag | `1` |
| VAMP analysis lags | `5`, `10`, `15` |
| Batch size reflected in outputs | `32`, e.g. `32x1001x2` |

**重要说明。** 这组结果具有科学参考价值，可评估 TITO 在 1 ns-step 长时步推理下的动力学表现；但它不是原文 Fig. 2 的严格同口径复现。

**Important note.** These results are scientifically useful for evaluating 1 ns-step TITO inference, but they are not a strict parameter-matched reproduction of Fig. 2.

## 4. 当前 VAMP-gap 结果 / Current VAMP-gap Results

**中文。** 全量 1251 个分子均完成。三组 `lag_vamp` 的统计如下：

| `lag_vamp` | mean gap | median gap | std | min | max | mean abs gap | median abs gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `-0.290921` | `0.005147` | `0.604663` | `-3.79686` | `0.990727` | `0.379364` | `0.098259` |
| 10 | `-0.227862` | `0.013140` | `0.574816` | `-3.66022` | `1.102260` | `0.345910` | `0.097032` |
| 15 | `-0.182827` | `0.020946` | `0.558528` | `-3.53096` | `1.109310` | `0.332613` | `0.103821` |

**English.** All 1251 molecules completed successfully. The sensitivity analysis across three `lag_vamp` values is:

| `lag_vamp` | mean gap | median gap | std | min | max | mean abs gap | median abs gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `-0.290921` | `0.005147` | `0.604663` | `-3.79686` | `0.990727` | `0.379364` | `0.098259` |
| 10 | `-0.227862` | `0.013140` | `0.574816` | `-3.66022` | `1.102260` | `0.345910` | `0.097032` |
| 15 | `-0.182827` | `0.020946` | `0.558528` | `-3.53096` | `1.109310` | `0.332613` | `0.103821` |

**解读。** `median gap` 在三个 lag 下均接近 0，说明典型分子的动力学差距较小；`median abs gap` 稳定在约 `0.097-0.104`，说明主结论对 `lag_vamp` 选择不敏感。signed mean 为负，说明结果被强负向尾部拉低。

**Interpretation.** The median gap is close to zero across all three lags, indicating small typical dynamical discrepancies. The median absolute gap remains stable around `0.097-0.104`, suggesting that the main conclusion is not sensitive to the chosen `lag_vamp`. The signed mean is negative because of a strong negative outlier tail.

## 5. 与原文结果对比 / Comparison with the Original Results

**中文。** 当前 `lag=1000 ps` 结果只能与原文作趋势对比，不能作严格数值对齐。原文小分子 Fig. 2 的 VAMP-2 gap mean/median 为 `-0.388 / -0.072`。我们当前 `lag=1000 ps, lag_vamp=10` 的 VAMP-gap mean/median 为 `-0.227862 / 0.013140`。

| 对比项 / Item | 原文 Fig. 2 小分子 / Original Fig. 2 Small Molecules | 当前 1 ns-step 复现 / Current 1 ns-step Reproduction | 判断 / Assessment |
|---|---:|---:|---|
| Sampling lag | `57 ps` | `1000 ps` | 不同 / different |
| Nested samples | `640` | `1000` | 不同 / different |
| ODE steps | `20` | `20` | 一致 / matched |
| VAMP-gap mean | `-0.388` | `-0.228` at `lag_vamp=10` | 趋势一致，数值不同 / trend consistent, numerically different |
| VAMP-gap median | `-0.072` | `0.013` at `lag_vamp=10` | 接近 0，但符号不同 / close to zero, sign differs |
| JSD mean / median | `0.097 / 0.087` | not computed | 未判定 / not determined |
| Top-10 relative error mean / median | `0.554 / 0.192` | not computed | 未判定 / not determined |

**English.** The current `lag=1000 ps` results can only be compared with the paper at the trend level, not as a strict parameter-matched comparison. The paper reports small-molecule Fig. 2 VAMP-2 gap mean/median of `-0.388 / -0.072`. Our current `lag=1000 ps, lag_vamp=10` VAMP-gap mean/median is `-0.227862 / 0.013140`.

**一致性结论。** VAMP-gap signed mean 均为负，因此方向上与原文“负 mean”的现象一致；但由于采样 lag、nested samples、以及未显式对齐的 VAMP estimator lag 不同，当前结果不能被报告为严格复现原文 Fig. 2。JSD 和 top-10 relative error 尚未计算，因此原文 Fig. 2 三指标完整复现仍未完成。

**Consistency conclusion.** The signed VAMP-gap mean is negative in both the paper and our reproduction, so the qualitative trend is consistent. However, because sampling lag, nested samples, and the VAMP estimator lag are not strictly matched, the current result should not be reported as a strict reproduction of Fig. 2. JSD and top-10 relative error remain uncomputed, so full reproduction of all three Fig. 2 metrics is incomplete.

## 6. Outliers / 异常分子

**中文。** 当前结果中最大绝对误差主要来自负向 outlier，说明这些分子上 TITO 的 VAMP score 明显高于 MD。稳定出现的负向 outlier 包括 `1198`, `1240`, `468`, `207`, `949/492`。稳定出现的正向 outlier 包括 `330`, `463`, `659`, `1035`。

**English.** The largest absolute errors are dominated by negative outliers, where TITO has a substantially higher VAMP score than MD. Stable negative outliers include `1198`, `1240`, `468`, `207`, and `949/492`. Stable positive outliers include `330`, `463`, `659`, and `1035`.

| 类别 / Category | 分子编号 / Molecule IDs | 解释 / Interpretation |
|---|---|---|
| Closest to zero | `517`, `1200`, `1189`, plus lag-specific examples | 典型成功样本 / typical successful cases |
| Largest negative gaps | `1198`, `1240`, `468`, `207`, `949/492` | TITO VAMP score 高于 MD；可能存在过慢动力学或新 metastable states / TITO VAMP score exceeds MD |
| Largest positive gaps | `330`, `463`, `659`, `1035` | TITO VAMP score 低于 MD；可能未捕捉 MD 慢过程 / TITO VAMP score below MD |

## 7. 原文 Fig. 2 对齐复现实验 / Fig. 2 Matched Reproduction Plan

**中文。** 为了和原文 Fig. 2 的 small molecules 统计口径严格对齐，需要补跑 `lag=57 ps` 的采样和分析。由于原文没有显式给出 `lag_vamp`，最佳当前推断是使用公开分析脚本默认值 `lag_vamp=1`，但报告中必须说明这是推断而非论文显式声明。

**English.** A strict Fig. 2 matched reproduction requires rerunning sampling and analysis at `lag=57 ps`. Because the paper does not explicitly specify `lag_vamp`, the current best inference is to use the public analysis script default, `lag_vamp=1`, while clearly stating that this is inferred rather than explicitly reported in the paper.

### Sampling / 采样

Dataset-based sampling is preferred for the full MDQM9 test split:

```bash
nohup python -u scripts/sample.py \
  --model_path inference_files/mdqm9.ckpt \
  --data_set mdqm9 \
  --data_path datasets/mdqm9-nc/ \
  --split test \
  --lag 57 \
  --nested_samples 640 \
  --batch_size 32 \
  --ode_steps 20 \
  --mol_indices $(seq 0 1250) \
  > logs/sample_fig2_lag57_nested640.log 2>&1 &
```

### Analysis / 分析

```bash
nohup python -u scripts/analyse.py \
  --data_set mdqm9 \
  --split test \
  --mol_indices $(seq 0 1250) \
  --model mdqm9 \
  --lag 57 \
  --nested_samples 640 \
  --ode_steps 20 \
  --lag_tica 1 \
  --lag_vamp 1 \
  > logs/analyse_fig2_lag57_vamp1.log 2>&1 &
```

### Summary / 汇总

```bash
python scripts/summarize_analysis.py \
  --data_set mdqm9 \
  --split test \
  --model mdqm9 \
  --lag 57 \
  --nested_samples 640 \
  --ode_steps 20 \
  --lag_vamp 1
```

## 8. 结论 / Conclusion

**中文。** 当前复现已经完成了 1 ns-step 长时步设置下的全量 VAMP-gap 分析，并显示典型分子的 VAMP-gap 接近 0，且对 `lag_vamp=5/10/15` 的选择较稳定。该结果与原文小分子 VAMP-gap signed mean 为负的趋势一致，但由于原文 Fig. 2 的 small-molecule sampling lag 是 `57 ps`，当前 `1000 ps` 设置不能作为严格同口径复现。严格复现需要补跑 `lag=57 ps, nested_samples=640, ode_steps=20, batch_size=32`，并进一步补齐 JSD 与 top-10 relative error。

**English.** The current reproduction completed full-test-set VAMP-gap analysis under a 1 ns-step setting. The typical molecule has a VAMP-gap close to zero, and the result is stable across `lag_vamp=5/10/15`. This is qualitatively consistent with the negative signed VAMP-gap mean reported for small molecules in the paper. However, because Fig. 2 uses `57 ps` sampling lag for small molecules, the current `1000 ps` setting is not a strict parameter-matched reproduction. A strict reproduction requires rerunning `lag=57 ps, nested_samples=640, ode_steps=20, batch_size=32` and adding JSD plus top-10 relative error.

## 9. Sources / 来源

- Diez, J. V., Schreiner, M., & Olsson, S. *Transferable generative models bridge femtosecond to nanosecond time-step molecular dynamics*. Science Advances 12, eaed2333 (2026). DOI: `10.1126/sciadv.aed2333`.
- Chalmers PDF: <https://research.chalmers.se/publication/551831/file/551831_Fulltext.pdf>
- arXiv page: <https://arxiv.org/abs/2510.07589>
- Current reproduction branch: `codex/tito-inference-fixes`

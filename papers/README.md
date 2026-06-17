# 论文清单（Papers）

本目录收录《预测模型技术详解报告》中引用的核心论文，便于离线阅读与截图配图。

> ⚠️ **说明**：本仓库的自动化环境开启了网络防火墙，**无法直接联网下载** `arxiv.org` / `bioinf.jku.at` 等学术站点的 PDF（域名解析被拦截）。
> 因此这里提供 **论文清单 + 一键下载脚本 `download_papers.sh`**，请在你本地（可联网的机器）执行脚本即可把所有 PDF 下载到本目录：
>
> ```bash
> cd papers
> bash download_papers.sh
> ```
>
> 若你希望由自动化环境直接下载并提交 PDF，请在仓库的 Copilot 设置中把 `arxiv.org`、`export.arxiv.org`、`www.bioinf.jku.at`、`proceedings.neurips.cc` 等域名加入允许列表后重试。

## 一、重点深读模型

| 文件名 | 论文 | 链接 |
| --- | --- | --- |
| `lstm_1997_hochreiter_schmidhuber.pdf` | LSTM（原始，Neural Computation 1997） | https://www.bioinf.jku.at/publications/older/2604.pdf |
| `lstm_forget_gate_gers_2000.pdf` | Learning to Forget（遗忘门，2000）* | https://doi.org/10.1162/089976600300015015 |
| `tabpfn_iclr2023_2207.01848.pdf` | TabPFN（ICLR 2023） | https://arxiv.org/abs/2207.01848 |
| `pfn_bayesian_inference_iclr2022_2112.10510.pdf` | Transformers Can Do Bayesian Inference（PFN 理论） | https://arxiv.org/abs/2112.10510 |
| `chronos_tmlr2024_2403.07815.pdf` | Chronos（TMLR 2024） | https://arxiv.org/abs/2403.07815 |

> \* TabPFN v2（Nature 2025, DOI: 10.1038/s41586-024-08328-6）与 LSTM 遗忘门（Neural Computation, MIT Press）为出版社版权页，通常需机构订阅，脚本不强制下载；可经 DOI 自行获取。

## 二、表格 / 集成学习

| 文件名 | 论文 | 链接 |
| --- | --- | --- |
| `xgboost_1603.02754.pdf` | XGBoost | https://arxiv.org/abs/1603.02754 |
| `lightgbm_neurips2017.pdf` | LightGBM | https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf |
| `catboost_1706.09516.pdf` | CatBoost | https://arxiv.org/abs/1706.09516 |
| `tabnet_1908.07442.pdf` | TabNet | https://arxiv.org/abs/1908.07442 |
| `ft_transformer_2106.11959.pdf` | FT-Transformer | https://arxiv.org/abs/2106.11959 |
| `dl_not_all_you_need_2106.03253.pdf` | Deep Learning is Not All You Need | https://arxiv.org/abs/2106.03253 |
| `grinsztajn_tree_vs_dl_2207.08815.pdf` | Why do tree-based models still outperform DL（Grinsztajn 2022） | https://arxiv.org/abs/2207.08815 |

## 三、时序预测

| 文件名 | 论文 | 链接 |
| --- | --- | --- |
| `informer_2012.07436.pdf` | Informer | https://arxiv.org/abs/2012.07436 |
| `autoformer_2106.13008.pdf` | Autoformer | https://arxiv.org/abs/2106.13008 |
| `fedformer_2201.12740.pdf` | FEDformer | https://arxiv.org/abs/2201.12740 |
| `patchtst_2211.14730.pdf` | PatchTST | https://arxiv.org/abs/2211.14730 |
| `itransformer_2310.06625.pdf` | iTransformer | https://arxiv.org/abs/2310.06625 |
| `dlinear_2205.13504.pdf` | DLinear（Are Transformers Effective?） | https://arxiv.org/abs/2205.13504 |
| `nbeats_1905.10437.pdf` | N-BEATS | https://arxiv.org/abs/1905.10437 |
| `nhits_2201.12886.pdf` | N-HiTS | https://arxiv.org/abs/2201.12886 |
| `tcn_1803.01271.pdf` | TCN | https://arxiv.org/abs/1803.01271 |
| `wavenet_1609.03499.pdf` | WaveNet | https://arxiv.org/abs/1609.03499 |
| `timesfm_2310.10688.pdf` | TimesFM | https://arxiv.org/abs/2310.10688 |

## 四、时序分类

| 文件名 | 论文 | 链接 |
| --- | --- | --- |
| `rocket_1910.13051.pdf` | ROCKET | https://arxiv.org/abs/1910.13051 |
| `minirocket_2012.08791.pdf` | MiniROCKET | https://arxiv.org/abs/2012.08791 |
| `inceptiontime_1909.04939.pdf` | InceptionTime | https://arxiv.org/abs/1909.04939 |

#!/usr/bin/env bash
# 一键下载《预测模型技术详解报告》引用的核心论文 PDF。
# 用法：在可联网的本地环境执行  ->  bash download_papers.sh
# 依赖：curl
#
# 注意：仓库的自动化（Copilot）环境默认开启网络防火墙，arxiv.org 等域名被拦截，
#       因此需要在你本地运行本脚本，或把相关域名加入仓库 Copilot 允许列表后再下载。

set -u
cd "$(dirname "$0")"

# arXiv 论文：文件名 <空格> arXiv-ID
ARXIV_PAPERS=(
  "tabpfn_iclr2023_2207.01848.pdf 2207.01848"
  "pfn_bayesian_inference_iclr2022_2112.10510.pdf 2112.10510"
  "chronos_tmlr2024_2403.07815.pdf 2403.07815"
  "xgboost_1603.02754.pdf 1603.02754"
  "catboost_1706.09516.pdf 1706.09516"
  "tabnet_1908.07442.pdf 1908.07442"
  "ft_transformer_2106.11959.pdf 2106.11959"
  "dl_not_all_you_need_2106.03253.pdf 2106.03253"
  "grinsztajn_tree_vs_dl_2207.08815.pdf 2207.08815"
  "informer_2012.07436.pdf 2012.07436"
  "autoformer_2106.13008.pdf 2106.13008"
  "fedformer_2201.12740.pdf 2201.12740"
  "patchtst_2211.14730.pdf 2211.14730"
  "itransformer_2310.06625.pdf 2310.06625"
  "dlinear_2205.13504.pdf 2205.13504"
  "nbeats_1905.10437.pdf 1905.10437"
  "nhits_2201.12886.pdf 2201.12886"
  "tcn_1803.01271.pdf 1803.01271"
  "wavenet_1609.03499.pdf 1609.03499"
  "timesfm_2310.10688.pdf 2310.10688"
  "rocket_1910.13051.pdf 1910.13051"
  "minirocket_2012.08791.pdf 2012.08791"
  "inceptiontime_1909.04939.pdf 1909.04939"
)

# 直链 PDF：文件名 <空格> URL
DIRECT_PAPERS=(
  "lstm_1997_hochreiter_schmidhuber.pdf https://www.bioinf.jku.at/publications/older/2604.pdf"
  "lightgbm_neurips2017.pdf https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf"
)

dl() { # dl <output> <url>
  local out="$1" url="$2"
  if [ -s "$out" ]; then
    echo "  [skip] $out 已存在"
    return 0
  fi
  echo "  [get ] $out  <-  $url"
  curl -fSL --retry 3 --connect-timeout 30 -o "$out" "$url" \
    && echo "  [ ok ] $out" \
    || echo "  [FAIL] $out （请检查网络/防火墙或手动从链接下载）"
}

echo "== 下载 arXiv 论文 =="
for entry in "${ARXIV_PAPERS[@]}"; do
  out="${entry%% *}"; id="${entry##* }"
  dl "$out" "https://arxiv.org/pdf/${id}"
done

echo "== 下载直链论文 =="
for entry in "${DIRECT_PAPERS[@]}"; do
  out="${entry%% *}"; url="${entry##* }"
  dl "$out" "$url"
done

echo "完成。出版社版权页（TabPFN v2 Nature 2025、LSTM 遗忘门 2000）通常需机构订阅，请经 DOI 自行获取。"

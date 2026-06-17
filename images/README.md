# 报告配图（images）

本目录下的 20 张 PNG 是《预测模型技术详解报告》正文的全部配图，全部由
`scripts/generate_figures.py` 使用 **matplotlib** 自动生成，均为**原创示意图**
（非论文截图），可随报告自由分发、修改、重绘。

## 重新生成

```bash
pip install matplotlib numpy scikit-learn statsmodels
# 中文显示需要 CJK 字体（如 Noto Sans CJK / 文泉驿等）
cd scripts
python generate_figures.py
```

脚本会把图片输出到本目录。若中文显示为方框，请安装任意中文字体并在
`scripts/fig_common.py` 的 `_CJK_CANDIDATES` 中补充字体路径。

## 图片清单

| 文件 | 对应 | 主题 |
| --- | --- | --- |
| `fig01_task_map.png` | 图1 | 预测任务技术地图（四象限+演进） |
| `fig02_tabular_tree.png` | 图2 | 表格模型技术演进树 |
| `fig03_logistic.png` | 图3 | 逻辑回归 Sigmoid 与决策边界 |
| `fig04_svm.png` | 图4 | SVM 最大间隔与支持向量 |
| `fig05_bagging_boosting.png` | 图5 | Bagging vs Boosting / GBDT 残差 |
| `fig06_tree_growth_hist.png` | 图6 | Level-wise vs Leaf-wise + 直方图分桶 |
| `fig07_tabnet_ft.png` | 图7 | TabNet / FT-Transformer 架构 |
| `fig08_tabpfn.png` | 图8 | TabPFN 工作流 |
| `fig09_ts_timeline.png` | 图9 | 时序技术演进时间轴 |
| `fig10_acf_decompose.png` | 图10 | ACF/PACF + 时序分解 |
| `fig11_sliding_window.png` | 图11 | 滑动窗口构造监督样本 |
| `fig12_rnn_unfold.png` | 图12 | RNN 展开 + 梯度消失 |
| `fig13_lstm.png` | 图13 | LSTM 单元结构 |
| `fig14_dilated_conv.png` | 图14 | 因果/膨胀卷积感受野 |
| `fig15_transformer_family.png` | 图15 | Transformer 时序家族/注意力/Patch |
| `fig16_dlinear_nbeats.png` | 图16 | DLinear / N-BEATS / 零样本 |
| `fig17_chronos.png` | 图17 | Chronos 流程 |
| `fig18_dtw_rocket_shapelet.png` | 图18 | DTW / ROCKET / Shapelet |
| `fig19_model_selection.png` | 图19 | 模型选型决策流程图 |
| `fig20_ts_cv.png` | 图20 | 时间序列交叉验证 |

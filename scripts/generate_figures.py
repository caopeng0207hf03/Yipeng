"""Generate all 20 figures for 预测模型技术详解报告.md.

Run: python scripts/generate_figures.py
Outputs PNGs into images/.
"""
import numpy as np
import matplotlib.pyplot as plt
from fig_common import (save, box, arrow, BLUE, GREEN, ORANGE, RED, PURPLE, GREY,
                        LIGHT, LIGHT2, LIGHT3)


# ---------------------------------------------------------------------------
# 1. 预测任务技术地图（四象限 + 技术演进时间轴）
# ---------------------------------------------------------------------------
def fig01():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # quadrant axes
    ax.axhline(5, color="black", lw=1.2)
    ax.axvline(5, color="black", lw=1.2)
    ax.text(9.7, 5.15, "回归", fontsize=12, ha="right", color=GREY)
    ax.text(0.3, 5.15, "分类", fontsize=12, ha="left", color=GREY)
    ax.text(2.5, 9.6, "← 分类      任务类型      回归 →", fontsize=11, ha="center", color=GREY)
    ax.text(0.25, 2.5, "时序 ←   数据形态   → 表格", fontsize=11, va="center",
            rotation=90, color=GREY)
    quad = [
        (7.5, 7.5, "表格·回归", "线性回归 / GBDT\nXGBoost·LightGBM\nTabPFN", LIGHT),
        (2.5, 7.5, "表格·分类", "逻辑回归 / SVM\n随机森林 / GBDT\nFT-Transformer", LIGHT2),
        (7.5, 2.5, "时序·回归", "ARIMA / LSTM·TCN\nInformer·PatchTST\nTimesFM·Chronos", LIGHT3),
        (2.5, 2.5, "时序·分类", "DTW-kNN / ROCKET\nInceptionTime\nShapelet", "#F5EEF8"),
    ]
    for x, y, t, m, c in quad:
        ax.add_patch(plt.Rectangle((x-2.2, y-1.7), 4.4, 3.4, fc=c, ec="none", alpha=0.6, zorder=0))
        ax.text(x, y+1.2, t, fontsize=13, ha="center", fontweight="bold", color=BLUE)
        ax.text(x, y-0.2, m, fontsize=10, ha="center", va="center")
    # timeline
    ax.text(5, 0.55, "技术演进时间轴：1960s 线性/ARIMA  →  2000s 集成学习(RF/GBDT)  "
            "→  2017+ Transformer  →  2023+ 基础模型(TabPFN/Chronos/TimesFM)",
            fontsize=9.5, ha="center", color=RED,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FDEDEC", ec=RED))
    ax.set_title("图1  预测任务技术地图（四象限 + 技术演进）", fontsize=14, fontweight="bold")
    save(fig, "fig01_task_map.png")


# ---------------------------------------------------------------------------
# 2. 表格模型技术演进树
# ---------------------------------------------------------------------------
def fig02():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    nodes = {
        "线性回归/逻辑回归": (1.4, 3.5, LIGHT),
        "SVM": (3.3, 5.0, LIGHT),
        "决策树": (3.3, 2.0, LIGHT2),
        "随机森林\n(Bagging)": (5.4, 3.4, LIGHT2),
        "GBDT\n(Boosting)": (5.4, 1.0, LIGHT3),
        "XGBoost": (7.8, 2.4, LIGHT3),
        "LightGBM": (7.8, 1.0, LIGHT3),
        "CatBoost": (7.8, -0.0, LIGHT3),
        "TabNet": (10.2, 4.6, "#F5EEF8"),
        "FT-Transformer": (10.2, 3.4, "#F5EEF8"),
        "TabPFN\n(表格基础模型)": (12.2, 4.0, "#FADBD8"),
    }
    w, h = 1.7, 0.85
    centers = {}
    for name, (x, y, c) in nodes.items():
        centers[name] = box(ax, x, y, w, h, name, fc=c, fs=9.5)
    edges = [
        ("线性回归/逻辑回归", "SVM"), ("线性回归/逻辑回归", "决策树"),
        ("决策树", "随机森林\n(Bagging)"), ("决策树", "GBDT\n(Boosting)"),
        ("GBDT\n(Boosting)", "XGBoost"), ("GBDT\n(Boosting)", "LightGBM"),
        ("GBDT\n(Boosting)", "CatBoost"),
        ("随机森林\n(Bagging)", "TabNet"), ("XGBoost", "FT-Transformer"),
        ("FT-Transformer", "TabPFN\n(表格基础模型)"),
        ("CatBoost", "TabPFN\n(表格基础模型)"),
    ]
    for a, b in edges:
        x1, y1 = centers[a]; x2, y2 = centers[b]
        arrow(ax, (x1 + w/2, y1), (x2 - w/2, y2), color=GREY, lw=1.6)
    ax.text(7, 6.6, "更强非线性 / 自动特征交互 →   更少调参 / 预训练 →",
            fontsize=10, ha="center", color=RED)
    ax.set_title("图2  表格模型技术演进树", fontsize=14, fontweight="bold")
    save(fig, "fig02_tabular_tree.png")


# ---------------------------------------------------------------------------
# 3. 逻辑回归 Sigmoid 决策边界
# ---------------------------------------------------------------------------
def fig03():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    X = np.r_[rng.randn(60, 2) * 0.9 + [-1.5, -1.0],
              rng.randn(60, 2) * 0.9 + [1.7, 1.3]]
    y = np.r_[np.zeros(60), np.ones(60)]
    clf = LogisticRegression().fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # left: sigmoid curve
    z = np.linspace(-8, 8, 200)
    axes[0].plot(z, 1/(1+np.exp(-z)), color=BLUE, lw=2.5)
    axes[0].axhline(0.5, ls="--", color=GREY); axes[0].axvline(0, ls="--", color=GREY)
    axes[0].set_title(r"Sigmoid 函数  $\sigma(z)=1/(1+e^{-z})$")
    axes[0].set_xlabel(r"$z = w^{T}x + b$"); axes[0].set_ylabel("P(y=1)")
    axes[0].grid(alpha=0.3)
    # right: decision boundary
    xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 300), np.linspace(-4, 4, 300))
    prob = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    cs = axes[1].contourf(xx, yy, prob, levels=20, cmap="RdBu_r", alpha=0.7)
    axes[1].contour(xx, yy, prob, levels=[0.5], colors="k", linewidths=2)
    axes[1].scatter(X[y==0,0], X[y==0,1], c="#2166AC", edgecolor="w", label="类别 0")
    axes[1].scatter(X[y==1,0], X[y==1,1], c="#B2182B", edgecolor="w", label="类别 1")
    axes[1].set_title("决策边界（P=0.5 等高线）")
    axes[1].legend(loc="upper left")
    fig.colorbar(cs, ax=axes[1], label="P(y=1)")
    fig.suptitle("图3  逻辑回归：Sigmoid 与线性决策边界", fontsize=14, fontweight="bold")
    save(fig, "fig03_logistic.png")


# ---------------------------------------------------------------------------
# 4. SVM 最大间隔 + 支持向量
# ---------------------------------------------------------------------------
def fig04():
    from sklearn.svm import SVC
    rng = np.random.RandomState(3)
    X = np.r_[rng.randn(40, 2) * 0.7 + [-1.6, -1.2],
              rng.randn(40, 2) * 0.7 + [1.8, 1.4]]
    y = np.r_[np.zeros(40), np.ones(40)]
    clf = SVC(kernel="linear", C=1.0).fit(X, y)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(X[y==0,0], X[y==0,1], c="#2166AC", edgecolor="w", s=45, label="类别 -1")
    ax.scatter(X[y==1,0], X[y==1,1], c="#B2182B", edgecolor="w", s=45, label="类别 +1")
    xx, yy = np.meshgrid(np.linspace(-4, 4.5, 300), np.linspace(-3.5, 4, 300))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["grey", "black", "grey"],
               linestyles=["--", "-", "--"], linewidths=[1.5, 2.2, 1.5])
    sv = clf.support_vectors_
    ax.scatter(sv[:,0], sv[:,1], s=200, facecolors="none", edgecolors=GREEN,
               linewidths=2.2, label="支持向量")
    ax.text(2.6, -2.6, "实线: 决策面 $w^{T}x+b=0$\n虚线: 间隔边界 $\\pm1$\n间隔 $=2/\\Vert w\\Vert$ 最大化",
            fontsize=10, bbox=dict(boxstyle="round", fc="#FEF9E7", ec=ORANGE))
    ax.legend(loc="upper left"); ax.set_title("图4  SVM 最大间隔与支持向量",
                                               fontsize=14, fontweight="bold")
    save(fig, "fig04_svm.png")


# ---------------------------------------------------------------------------
# 5. Bagging vs Boosting + GBDT 残差拟合
# ---------------------------------------------------------------------------
def fig05():
    fig = plt.figure(figsize=(13, 5.5))
    # left schematic: bagging vs boosting
    ax = fig.add_subplot(1, 2, 1); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(2.5, 9.4, "Bagging（并行·投票）", ha="center", fontsize=12, fontweight="bold", color=GREEN)
    box(ax, 1.4, 7.6, 2.2, 0.9, "数据集", fc=LIGHT)
    for i, yy in enumerate([5.8, 4.4, 3.0]):
        box(ax, 0.4, yy, 1.8, 0.9, f"子样本{i+1}", fc=LIGHT2, fs=9)
        box(ax, 2.7, yy, 1.7, 0.9, f"树{i+1}", fc=LIGHT2, fs=9)
        arrow(ax, (2.2, yy+0.45), (2.7, yy+0.45))
        arrow(ax, (2.5, 7.6), (1.3, yy+0.9), color=GREY, lw=1)
    box(ax, 1.3, 1.2, 2.4, 0.9, "投票/平均", fc=LIGHT3)
    for yy in [5.8, 4.4, 3.0]:
        arrow(ax, (3.5, yy), (2.6, 2.1), color=GREY, lw=1)
    # right of same subplot: boosting
    ax.text(7.5, 9.4, "Boosting（串行·纠错）", ha="center", fontsize=12, fontweight="bold", color=ORANGE)
    prev = None
    for i, xx in enumerate([5.6, 7.2, 8.8]):
        c = box(ax, xx-0.7, 4.6, 1.5, 0.9, f"树{i+1}", fc=LIGHT3, fs=9)
        ax.text(xx, 3.9, "拟合残差", fontsize=8, ha="center", color=RED)
        if prev: arrow(ax, prev, (xx-0.7, 5.05))
        prev = (xx+0.8, 5.05)
    ax.text(7.5, 6.2, "每棵树纠正前序累计误差", fontsize=9, ha="center", color=GREY)
    # right subplot: GBDT residual fitting
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.linspace(0, 6, 120)
    f = np.sin(x) + 0.3*x
    pred = np.zeros_like(x); lr = 0.5
    rng = np.random.RandomState(1); ynoise = f + rng.randn(len(x))*0.12
    ax2.plot(x, ynoise, ".", color=GREY, ms=4, label="目标(含噪)")
    colors = ["#FADBD8", "#F1948A", "#CB4335"]
    for k, c in enumerate(colors):
        # crude stage-wise fit using binned means of residual
        res = ynoise - pred
        bins = np.linspace(0, 6, 6 + k*4)
        idx = np.digitize(x, bins)
        step = np.array([res[idx==b].mean() if np.any(idx==b) else 0 for b in idx])
        pred = pred + lr*step
        ax2.plot(x, pred, color=c, lw=2, label=f"累计 {k+1} 棵树")
    ax2.plot(x, f, "k--", lw=1.5, label="真实函数")
    ax2.legend(fontsize=8); ax2.set_title("GBDT 逐步拟合残差")
    fig.suptitle("图5  Bagging vs Boosting 与 GBDT 残差拟合", fontsize=14, fontweight="bold")
    save(fig, "fig05_bagging_boosting.png")


# ---------------------------------------------------------------------------
# 6. Level-wise vs Leaf-wise + 直方图分桶
# ---------------------------------------------------------------------------
def fig06():
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Level-wise (XGBoost) vs Leaf-wise (LightGBM)", fontsize=11)
    def node(ax, x, y, c=BLUE, r=0.32):
        ax.add_patch(plt.Circle((x, y), r, fc=c, ec="black", zorder=3))
    # level-wise (balanced)
    ax.text(2.5, 9.2, "Level-wise（按层）", ha="center", color=GREEN, fontsize=10)
    lvl = {(2.5,8):None,(1.3,6.5):(2.5,8),(3.7,6.5):(2.5,8),
           (0.7,5):(1.3,6.5),(1.9,5):(1.3,6.5),(3.1,5):(3.7,6.5),(4.3,5):(3.7,6.5)}
    for (x,y),p in lvl.items():
        node(ax,x,y,GREEN)
        if p: arrow(ax,p,(x,y),color=GREY,lw=1.2)
    # leaf-wise (deep on best leaf)
    ax.text(7.5, 9.2, "Leaf-wise（按最大增益叶）", ha="center", color=ORANGE, fontsize=10)
    leaf = {(7.5,8):None,(6.6,6.6):(7.5,8),(8.4,6.6):(7.5,8),
            (6.0,5.2):(6.6,6.6),(7.2,5.2):(6.6,6.6),
            (6.6,3.8):(7.2,5.2),(7.8,3.8):(7.2,5.2)}
    for (x,y),p in leaf.items():
        node(ax,x,y,ORANGE)
        if p: arrow(ax,p,(x,y),color=GREY,lw=1.2)
    ax.text(5, 1.2, "Level-wise 平衡、抗过拟合；Leaf-wise 更深、精度高但需 max_depth 约束",
            ha="center", fontsize=9, color=GREY)
    # histogram binning
    ax2 = fig.add_subplot(1, 2, 2)
    rng = np.random.RandomState(2)
    vals = np.r_[rng.normal(3,1,400), rng.normal(7,1.2,300)]
    counts, edges, patches = ax2.hist(vals, bins=16, color=LIGHT, edgecolor=BLUE)
    ax2.set_title("Histogram 直方图算法：连续特征分桶")
    ax2.set_xlabel("特征值 → 分到 bin"); ax2.set_ylabel("样本数")
    ax2.axvline(5, color=RED, ls="--", lw=2)
    ax2.text(5.1, counts.max()*0.9, "分裂只需遍历 bin 边界\n→ 降内存/加速", color=RED, fontsize=9)
    fig.suptitle("图6  树生长策略与直方图分桶", fontsize=14, fontweight="bold")
    save(fig, "fig06_tree_growth_hist.png")


# ---------------------------------------------------------------------------
# 7. TabNet / FT-Transformer 架构
# ---------------------------------------------------------------------------
def fig07():
    fig = plt.figure(figsize=(13, 5.8))
    ax = fig.add_subplot(1, 2, 1); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("TabNet（序贯注意力特征选择）", fontsize=11, color=BLUE)
    box(ax, 3, 8.6, 4, 0.9, "输入特征", fc=LIGHT)
    prev=(5,8.6)
    ys=[6.9,5.0,3.1]
    for i,y in enumerate(ys):
        c=box(ax,1,y,3.3,1.1,f"Step {i+1}\nAttentive\nTransformer\n(特征掩码)",fc=LIGHT2,fs=8)
        d=box(ax,5.7,y,3.3,1.1,f"Feature\nTransformer\n→ 决策输出",fc=LIGHT3,fs=8)
        arrow(ax,(4.3,y+0.55),(5.7,y+0.55))
        arrow(ax,prev,(2.6,y+1.1),color=GREY,lw=1.2)
        prev=(2.6,y)
    box(ax,3.2,1.2,3.6,0.9,"加和 → 预测",fc="#FADBD8")
    for y in ys: arrow(ax,(7.3,y),(5.4,2.1),color=GREY,lw=1)
    ax.text(5,0.4,"可学习的稀疏特征选择 → 可解释",ha="center",fontsize=8,color=GREY)
    # FT-Transformer
    ax2 = fig.add_subplot(1, 2, 2); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("FT-Transformer（特征→Token→Transformer）", fontsize=11, color=PURPLE)
    box(ax2,0.5,8.6,4,0.8,"数值特征",fc=LIGHT); box(ax2,5.5,8.6,4,0.8,"类别特征",fc=LIGHT)
    box(ax2,1.5,7.0,7,0.9,"Feature Tokenizer（每个特征→嵌入向量）",fc=LIGHT2,fs=9)
    arrow(ax2,(2.5,8.6),(4,7.9)); arrow(ax2,(7.5,8.6),(6,7.9))
    box(ax2,0.6,5.7,1.5,0.7,"[CLS]",fc="#FADBD8",fs=9)
    box(ax2,2.4,5.7,6,0.7,"特征 token 序列",fc=LIGHT2,fs=9)
    arrow(ax2,(5,7.0),(5,6.4))
    box(ax2,1.5,3.6,7,1.3,"Transformer Encoder\n(多头自注意力 × L 层)",fc=LIGHT3,fs=10)
    arrow(ax2,(5,5.7),(5,4.9))
    box(ax2,2.8,1.6,4.4,0.9,"取 [CLS] → MLP → 预测",fc="#FADBD8",fs=9)
    arrow(ax2,(5,3.6),(5,2.5))
    fig.suptitle("图7  TabNet 与 FT-Transformer 架构示意", fontsize=14, fontweight="bold")
    save(fig, "fig07_tabnet_ft.png")


# ---------------------------------------------------------------------------
# 8. TabPFN 工作流
# ---------------------------------------------------------------------------
def fig08():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    box(ax,0.4,3.0,2.6,1.4,"① 合成先验\n从结构因果模型\n采样大量人工数据集",fc=LIGHT,fs=9)
    box(ax,3.6,3.0,2.6,1.4,"② 一次性元训练\nTransformer 学会\n“如何在表格上学习”",fc=LIGHT2,fs=9)
    box(ax,6.8,3.0,2.6,1.4,"③ 部署：单次前向\nin-context 预测\n(无需梯度训练)",fc=LIGHT3,fs=9)
    box(ax,10.0,3.0,1.7,1.4,"④ 输出\n后验预测分布",fc="#FADBD8",fs=9)
    arrow(ax,(3.0,3.7),(3.6,3.7),color=BLUE,lw=2)
    arrow(ax,(6.2,3.7),(6.8,3.7),color=BLUE,lw=2)
    arrow(ax,(9.4,3.7),(10.0,3.7),color=BLUE,lw=2)
    box(ax,5.0,0.6,4,1.0,"训练集 + 测试样本\n一起作为上下文输入",fc=LIGHT,fs=9)
    arrow(ax,(7,1.6),(8.0,3.0),color=GREEN,lw=1.6)
    ax.text(6,6.2,"核心：把“训练”变成一次前向传播（贝叶斯推断的摊还）",
            ha="center",fontsize=11,color=RED)
    ax.set_title("图8  TabPFN 工作流：合成先验 → 元训练 → in-context 预测",
                 fontsize=14, fontweight="bold")
    save(fig, "fig08_tabpfn.png")


# ---------------------------------------------------------------------------
# 9. 时序技术演进时间轴
# ---------------------------------------------------------------------------
def fig09():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    ax.axhline(3, color="black", lw=2)
    items = [
        (1.0, "1970s", "ARIMA\n统计模型", LIGHT, True),
        (3.0, "1997", "LSTM\n门控RNN", LIGHT2, False),
        (5.0, "2016+", "WaveNet/TCN\n膨胀卷积", LIGHT3, True),
        (7.2, "2017", "Transformer\n自注意力", LIGHT, False),
        (9.2, "2021+", "Informer/Autoformer\nPatchTST/iTransformer", LIGHT2, True),
        (11.4, "2023+", "TimesFM/Chronos\n时序基础模型", "#FADBD8", False),
    ]
    for x, yr, t, c, up in items:
        ax.plot(x, 3, "o", color=BLUE, ms=10, zorder=3)
        y = 4.2 if up else 1.0
        box(ax, x-0.95, y, 1.9, 1.2, t, fc=c, fs=9)
        ax.plot([x, x], [3, y+1.2 if not up else y], color=GREY, lw=1)
        ax.text(x, 3.35 if up else 2.45, yr, ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.text(6.5, 5.6, "统计 → 循环 → 卷积 → 注意力 → 预训练基础模型",
            ha="center", fontsize=11, color=GREY)
    ax.set_title("图9  时序预测技术演进时间轴", fontsize=14, fontweight="bold")
    save(fig, "fig09_ts_timeline.png")


# ---------------------------------------------------------------------------
# 10. ACF/PACF + 时序分解
# ---------------------------------------------------------------------------
def fig10():
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    rng = np.random.RandomState(0)
    n = 240
    t = np.arange(n)
    trend = 0.03 * t
    season = 2.0 * np.sin(2 * np.pi * t / 12)
    noise = rng.randn(n) * 0.5
    y = 5 + trend + season + noise
    import pandas as pd
    s = pd.Series(y, index=pd.period_range("2005-01", periods=n, freq="M").to_timestamp())
    fig = plt.figure(figsize=(13, 7))
    ax1 = fig.add_subplot(2, 2, 1); plot_acf(y, ax=ax1, lags=36); ax1.set_title("ACF（自相关）")
    ax2 = fig.add_subplot(2, 2, 2); plot_pacf(y, ax=ax2, lags=36, method="ywm"); ax2.set_title("PACF（偏自相关）")
    dec = seasonal_decompose(s, model="additive", period=12)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(s.index, dec.trend, color=ORANGE); ax3.set_title("趋势 Trend"); ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(s.index, dec.seasonal, color=GREEN); ax4.set_title("季节 Seasonal"); ax4.grid(alpha=0.3)
    fig.suptitle("图10  ACF/PACF 定阶 与 时序分解（趋势/季节）", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig10_acf_decompose.png")


# ---------------------------------------------------------------------------
# 11. 滑动窗口 → 监督样本构造
# ---------------------------------------------------------------------------
def fig11():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    series = [3.0,3.4,3.1,3.8,4.2,4.0,4.6,5.1,4.9,5.5]
    for i, v in enumerate(series):
        ax.add_patch(plt.Rectangle((i*1.2+0.5, 6.4), 1.0, 0.9, fc=LIGHT, ec=BLUE))
        ax.text(i*1.2+1.0, 6.85, f"{v}", ha="center", va="center", fontsize=9)
        ax.text(i*1.2+1.0, 7.55, f"t{i+1}", ha="center", fontsize=8, color=GREY)
    ax.text(7, 8.0, "原始时间序列", ha="center", fontsize=11, color=GREY)
    # three windows
    rows = [(0, 5.2, RED, "窗口1: [t1..t3] → t4"),
            (1, 4.0, ORANGE, "窗口2: [t2..t4] → t5"),
            (2, 2.8, GREEN, "窗口3: [t3..t5] → t6")]
    for k, y, c, lab in rows:
        for j in range(3):
            xi = (k+j)
            ax.add_patch(plt.Rectangle((xi*1.2+0.5, y), 1.0, 0.8, fc=LIGHT2, ec=c, lw=2))
            ax.text(xi*1.2+1.0, y+0.4, f"{series[k+j]}", ha="center", va="center", fontsize=9)
        tx = (k+3)
        ax.add_patch(plt.Rectangle((tx*1.2+0.5, y), 1.0, 0.8, fc="#FADBD8", ec=c, lw=2))
        ax.text(tx*1.2+1.0, y+0.4, f"{series[k+3]}", ha="center", va="center", fontsize=9)
        ax.text(9.6, y+0.4, lab, fontsize=10, color=c, va="center")
    ax.text(7, 1.3, "特征 X = 过去 L 步（look-back）   |   标签 y = 未来 H 步（horizon）",
            ha="center", fontsize=11, color=BLUE,
            bbox=dict(boxstyle="round", fc="#FEF9E7", ec=ORANGE))
    ax.set_title("图11  滑动窗口构造监督学习样本", fontsize=14, fontweight="bold")
    save(fig, "fig11_sliding_window.png")


# ---------------------------------------------------------------------------
# 12. RNN 按时间展开 + 梯度消失
# ---------------------------------------------------------------------------
def fig12():
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("RNN 按时间展开（unfold）", fontsize=11)
    prev = None
    for i, x in enumerate([1.5, 4.5, 7.5, 10.5]):
        c = box(ax, x-0.8, 2.6, 1.6, 1.1, f"h{i}", fc=LIGHT2, fs=11)
        box(ax, x-0.6, 0.6, 1.2, 0.8, f"x{i}", fc=LIGHT, fs=10)
        box(ax, x-0.6, 4.6, 1.2, 0.8, f"y{i}", fc=LIGHT3, fs=10)
        arrow(ax, (x, 1.4), (x, 2.6))
        arrow(ax, (x, 3.7), (x, 4.6))
        if prev: arrow(ax, prev, (x-0.8, 3.15), color=BLUE, lw=2)
        prev = (x+0.8, 3.15)
    ax.text(6, 5.7, "同一组权重 W 在每个时刻复用", ha="center", fontsize=9, color=GREY)
    # gradient vanishing intuition
    ax2 = fig.add_subplot(1, 2, 2)
    k = np.arange(0, 30)
    ax2.semilogy(k, 0.7**k, color=RED, lw=2, label=r"$|\partial|\approx0.7^{k}$ 梯度消失")
    ax2.semilogy(k, 1.15**k, color=ORANGE, lw=2, ls="--", label=r"$|\partial|\approx1.15^{k}$ 梯度爆炸")
    ax2.axhline(1, color=GREY, ls=":")
    ax2.set_xlabel("时间步距离 k（回传步数）"); ax2.set_ylabel("梯度幅度(对数)")
    ax2.set_title("长程依赖下的梯度消失/爆炸"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("图12  RNN 展开与梯度消失直觉", fontsize=14, fontweight="bold")
    save(fig, "fig12_rnn_unfold.png")


# ---------------------------------------------------------------------------
# 13. LSTM 单元结构 + 梯度流
# ---------------------------------------------------------------------------
def fig13():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
    # cell state highway
    arrow(ax, (0.6, 7.4), (11.4, 7.4), color=RED, lw=3)
    ax.text(6, 7.8, "细胞状态 $C_t$ (CEC 恒定误差流 → 缓解梯度消失)", ha="center",
            fontsize=10, color=RED)
    # gates
    box(ax, 1.6, 4.2, 2.0, 1.1, "遗忘门 $f_t$\n$\\sigma$", fc=LIGHT, fs=10)
    box(ax, 4.4, 4.2, 2.0, 1.1, "输入门 $i_t$\n$\\sigma$ + tanh", fc=LIGHT2, fs=10)
    box(ax, 7.6, 4.2, 2.0, 1.1, "输出门 $o_t$\n$\\sigma$", fc=LIGHT3, fs=10)
    # operators on highway
    for x, sym, c in [(2.6, "×", ORANGE), (5.4, "+", GREEN), (9.2, "×", PURPLE)]:
        ax.add_patch(plt.Circle((x, 7.4), 0.28, fc="white", ec=c, lw=2, zorder=4))
        ax.text(x, 7.4, sym, ha="center", va="center", fontsize=13, color=c, zorder=5)
        arrow(ax, (x, 5.3), (x, 7.1), color=c, lw=1.5)
    box(ax, 0.4, 1.2, 1.8, 0.9, "$h_{t-1},x_t$", fc=LIGHT, fs=10)
    for x in [2.6, 5.4, 8.6]:
        arrow(ax, (1.5, 2.1), (x-0.4, 4.2), color=GREY, lw=1)
    box(ax, 10.2, 4.2, 1.4, 1.1, "$h_t$\ntanh", fc="#FADBD8", fs=10)
    arrow(ax, (9.2, 7.1), (10.9, 5.3), color=PURPLE, lw=1.5)
    ax.text(6, 0.4, "关键公式：$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$  （加性更新是梯度稳定的核心）",
            ha="center", fontsize=10, color=BLUE,
            bbox=dict(boxstyle="round", fc="#FEF9E7", ec=ORANGE))
    ax.set_title("图13  LSTM 单元结构（细胞状态 + 三门控）", fontsize=14, fontweight="bold")
    save(fig, "fig13_lstm.png")


# ---------------------------------------------------------------------------
# 14. 因果卷积 + 膨胀卷积感受野
# ---------------------------------------------------------------------------
def fig14():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(-0.5, 16); ax.set_ylim(0, 9); ax.axis("off")
    layers = 4
    dilations = [1, 2, 4, 8]
    n = 16
    ypos = [1, 3, 5, 7]
    # nodes
    coords = {}
    for li, y in enumerate(ypos):
        for i in range(n):
            coords[(li, i)] = (i, y)
            ax.add_patch(plt.Circle((i, y), 0.16, fc=LIGHT, ec=BLUE, zorder=3))
    labels = ["输入", "d=1", "d=2", "d=4"]
    for li, y in enumerate(ypos):
        ax.text(-0.5, y, labels[li], fontsize=9, ha="right", va="center", color=GREY)
    # causal dilated connections feeding into the last output node, highlight receptive field
    hi = n - 1
    rf = set([hi])
    for li in range(1, layers):
        d = 2 ** (li - 1)
        new = set()
        for node in list(rf):
            for src in (node, node - d):
                if 0 <= src:
                    ax.plot([coords[(li-1, src)][0], coords[(li, node)][0]],
                            [coords[(li-1, src)][1], coords[(li, node)][1]],
                            color=RED, lw=1.3, zorder=2)
                    new.add(src)
        rf = new
    # recolor receptive-field input nodes
    for i in rf:
        ax.add_patch(plt.Circle(coords[(0, i)], 0.16, fc="#FADBD8", ec=RED, zorder=4))
    ax.add_patch(plt.Circle(coords[(3, hi)], 0.2, fc="#FADBD8", ec=RED, zorder=4))
    ax.text(7.5, 8.4, "因果膨胀卷积：感受野随层数指数增长 ($2^{L}$)，且只看过去（因果）",
            ha="center", fontsize=11, color=RED)
    ax.set_title("图14  因果卷积与膨胀卷积感受野", fontsize=14, fontweight="bold")
    save(fig, "fig14_dilated_conv.png")


# ---------------------------------------------------------------------------
# 15. Transformer 时序家族 + 自注意力热力图 + Patch 切分
# ---------------------------------------------------------------------------
def fig15():
    fig = plt.figure(figsize=(14, 5))
    # family tree
    ax = fig.add_subplot(1, 3, 1); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Transformer 时序家族", fontsize=11)
    box(ax, 3, 8.6, 4, 0.9, "Transformer", fc=LIGHT)
    fam = [("Informer\n(ProbSparse)", 1.0, 6.6, LIGHT2),
           ("Autoformer\n(分解+自相关)", 5.2, 6.6, LIGHT2),
           ("FEDformer\n(频域)", 1.0, 4.4, LIGHT3),
           ("PatchTST\n(分块)", 5.2, 4.4, LIGHT3),
           ("iTransformer\n(变量为token)", 3.0, 2.2, "#FADBD8")]
    for t, x, y, c in fam:
        box(ax, x, y, 3.6, 1.0, t, fc=c, fs=8.5)
        arrow(ax, (5, 8.6), (x+1.8, y+1.0), color=GREY, lw=1)
    # attention heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    rng = np.random.RandomState(5)
    A = rng.rand(12, 12)
    A = A / A.sum(1, keepdims=True)
    im = ax2.imshow(A, cmap="viridis")
    ax2.set_title("自注意力权重热力图"); ax2.set_xlabel("Key 位置"); ax2.set_ylabel("Query 位置")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    # patch split
    ax3 = fig.add_subplot(1, 3, 3)
    x = np.linspace(0, 6, 96)
    y = np.sin(x) + 0.4*np.sin(3*x)
    ax3.plot(x, y, color=BLUE)
    for s in range(0, 96, 16):
        ax3.axvspan(x[s], x[min(s+15, 95)], color=ORANGE, alpha=0.12)
        ax3.axvline(x[s], color=ORANGE, ls="--", lw=0.8)
    ax3.set_title("PatchTST：序列切分为 patch token")
    ax3.set_xlabel("时间"); ax3.grid(alpha=0.3)
    fig.suptitle("图15  Transformer 时序家族 / 自注意力 / Patch 切分", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig15_transformer_family.png")


# ---------------------------------------------------------------------------
# 16. DLinear / N-BEATS / 基础模型零样本
# ---------------------------------------------------------------------------
def fig16():
    fig = plt.figure(figsize=(14, 5))
    # DLinear
    ax = fig.add_subplot(1, 3, 1); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("DLinear（分解 + 线性）", fontsize=11)
    box(ax, 3, 8.6, 4, 0.9, "输入序列", fc=LIGHT)
    box(ax, 0.6, 6.0, 4, 1.0, "移动平均\n→ 趋势项", fc=LIGHT2, fs=9)
    box(ax, 5.4, 6.0, 4, 1.0, "残差\n→ 季节项", fc=LIGHT3, fs=9)
    arrow(ax, (4, 8.6), (2.6, 7.0)); arrow(ax, (6, 8.6), (7.4, 7.0))
    box(ax, 0.6, 3.6, 4, 0.9, "线性层", fc=LIGHT); box(ax, 5.4, 3.6, 4, 0.9, "线性层", fc=LIGHT)
    arrow(ax, (2.6, 6.0), (2.6, 4.5)); arrow(ax, (7.4, 6.0), (7.4, 4.5))
    box(ax, 3, 1.4, 4, 0.9, "相加 → 预测", fc="#FADBD8")
    arrow(ax, (2.6, 3.6), (4.5, 2.3)); arrow(ax, (7.4, 3.6), (5.5, 2.3))
    # N-BEATS
    ax2 = fig.add_subplot(1, 3, 2); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("N-BEATS（堆叠残差块）", fontsize=11)
    prev = (5, 9.2)
    for i, y in enumerate([7.2, 4.8, 2.4]):
        box(ax2, 2.4, y, 5.2, 1.4, f"Block {i+1}\nFC → backcast / forecast", fc=LIGHT2, fs=9)
        arrow(ax2, prev, (5, y+1.4), color=BLUE, lw=2)
        ax2.text(8.0, y+0.7, "forecast", fontsize=8, color=GREEN)
        arrow(ax2, (7.6, y+0.4), (9.2, y+0.4), color=GREEN, lw=1.5)
        prev = (5, y)
    ax2.text(5, 0.9, "残差逐块剥离 → 累加 forecast", ha="center", fontsize=9, color=GREY)
    # zero-shot foundation model
    ax3 = fig.add_subplot(1, 3, 3)
    x = np.arange(60)
    rng = np.random.RandomState(7)
    hist = 10 + 2*np.sin(x/4) + rng.randn(60)*0.3
    fx = np.arange(60, 80)
    fut = 10 + 2*np.sin(fx/4)
    ax3.plot(x, hist, color=BLUE, label="历史(context)")
    ax3.plot(fx, fut, color=RED, ls="--", lw=2, label="零样本预测")
    ax3.fill_between(fx, fut-0.8, fut+0.8, color=RED, alpha=0.15)
    ax3.axvline(60, color=GREY, ls=":")
    ax3.set_title("时序基础模型：零样本预测"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    fig.suptitle("图16  DLinear / N-BEATS / 基础模型零样本", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig16_dlinear_nbeats.png")


# ---------------------------------------------------------------------------
# 17. Chronos 流程
# ---------------------------------------------------------------------------
def fig17():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    box(ax,0.4,3.0,2.4,1.5,"① 原始时序\n实数序列",fc=LIGHT,fs=9)
    box(ax,3.2,3.0,2.4,1.5,"② Scaling\n均值/尺度归一化",fc=LIGHT2,fs=9)
    box(ax,6.0,3.0,2.4,1.5,"③ Quantization\n离散化为 token\n(类比词表)",fc=LIGHT3,fs=9)
    box(ax,8.8,3.0,2.4,1.5,"④ 语言模型\nT5/Transformer\n预测下一 token",fc="#F5EEF8",fs=9)
    box(ax,11.0,3.0,1.7,1.5,"⑤ 反量化\n→ 概率预测",fc="#FADBD8",fs=9)
    for x in [2.8,5.6,8.4,10.6]:
        arrow(ax,(x,3.75),(x+0.4,3.75),color=BLUE,lw=2)
    ax.text(6.5,6.0,"把时序预测当作“语言建模”：序列 → token → 自回归生成",
            ha="center",fontsize=11,color=RED)
    ax.text(6.5,1.2,"训练在大规模真实+合成时序语料上完成，支持零样本预测",
            ha="center",fontsize=9.5,color=GREY)
    ax.set_title("图17  Chronos：scaling → quantization → 语言模型 token 预测",
                 fontsize=13.5, fontweight="bold")
    save(fig, "fig17_chronos.png")


# ---------------------------------------------------------------------------
# 18. DTW 对齐 + ROCKET 随机卷积核 + Shapelet
# ---------------------------------------------------------------------------
def fig18():
    fig = plt.figure(figsize=(14, 5))
    # DTW alignment
    ax = fig.add_subplot(1, 3, 1)
    t = np.linspace(0, 6, 50)
    a = np.sin(t)
    b = np.sin(t - 0.8) + 0.15
    ax.plot(t, a + 1.5, color=BLUE, lw=2, label="序列 A")
    ax.plot(t, b - 1.5, color=ORANGE, lw=2, label="序列 B")
    for i in range(0, 50, 5):
        # nearest warping (simple offset) just to illustrate alignment lines
        j = min(49, i + 6)
        ax.plot([t[i], t[j]], [a[i] + 1.5, b[j] - 1.5], color=GREY, lw=0.7)
    ax.set_title("DTW 动态时间规整对齐"); ax.legend(fontsize=8); ax.axis("off")
    # ROCKET random kernels
    ax2 = fig.add_subplot(1, 3, 2)
    rng = np.random.RandomState(0)
    x = np.linspace(0, 6, 120)
    sig = np.sin(x) + 0.3 * np.sin(4 * x)
    ax2.plot(x, sig, color=BLUE, lw=2)
    for off, c in [(0.0, RED), (0.0, GREEN), (0.0, PURPLE)]:
        k = rng.randn(9)
        conv = np.convolve(sig, k, mode="same") / 6
        ax2.plot(x, conv + 0, color=c, lw=1, alpha=0.7)
    ax2.set_title("ROCKET：大量随机卷积核\n→ PPV/Max 特征 → 线性分类器", fontsize=10)
    ax2.set_xlabel("时间"); ax2.grid(alpha=0.3)
    # Shapelet
    ax3 = fig.add_subplot(1, 3, 3)
    x = np.linspace(0, 10, 200)
    series = np.sin(x) + 0.2 * rng.randn(200)
    ax3.plot(x, series, color=GREY, lw=1.2, label="时间序列")
    mask = (x > 4) & (x < 5.6)
    ax3.plot(x[mask], series[mask], color=RED, lw=3, label="判别性子序列 Shapelet")
    ax3.set_title("Shapelet：最具判别力的局部形状"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    fig.suptitle("图18  时序分类：DTW / ROCKET / Shapelet", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig18_dtw_rocket_shapelet.png")


# ---------------------------------------------------------------------------
# 19. 模型选型决策流程图
# ---------------------------------------------------------------------------
def fig19():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")
    def dia(x, y, w, h, text, c=LIGHT3):
        ax.add_patch(plt.Polygon([(x, y+h/2), (x+w/2, y+h), (x+w, y+h/2), (x+w/2, y)],
                                 fc=c, ec=ORANGE, lw=1.5, zorder=2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9, zorder=3)
        return (x+w/2, y+h/2)
    d1 = dia(4.5, 9.2, 3, 1.4, "数据类型？")
    ax.text(2.2, 9.9, "表格", color=GREEN, fontsize=9); ax.text(9.4, 9.9, "时序", color=BLUE, fontsize=9)
    # tabular branch
    d2 = dia(0.6, 6.6, 3, 1.4, "数据量大且需\n多模态/迁移？")
    arrow(ax, (5.0, 9.2), (2.1, 8.0))
    b_t1 = box(ax, 0.4, 4.2, 3.4, 1.0, "GBDT\n(XGB/LGBM/CatBoost)", fc=LIGHT2, fs=9)
    b_t2 = box(ax, 0.4, 2.4, 3.4, 1.0, "深度表格 / TabPFN\n(小样本可零样本)", fc="#FADBD8", fs=9)
    arrow(ax, (1.5, 6.6), (1.6, 5.2)); ax.text(1.7, 5.9, "否", fontsize=8, color=GREEN)
    arrow(ax, (2.1, 6.6), (2.6, 3.4)); ax.text(2.9, 5.9, "是", fontsize=8, color=RED)
    # ts branch
    d3 = dia(8.4, 6.6, 3.2, 1.4, "长程/多变量\n或需零样本？")
    arrow(ax, (7.0, 9.2), (10.0, 8.0))
    b_s1 = box(ax, 8.2, 4.2, 3.6, 1.0, "经典/轻量\nARIMA·DLinear·LSTM", fc=LIGHT2, fs=9)
    b_s2 = box(ax, 8.2, 2.4, 3.6, 1.0, "Transformer 家族 /\n基础模型(Chronos/TimesFM)", fc="#FADBD8", fs=8.5)
    arrow(ax, (9.4, 6.6), (9.4, 5.2)); ax.text(9.5, 5.9, "否", fontsize=8, color=GREEN)
    arrow(ax, (10.4, 6.6), (10.2, 3.4)); ax.text(10.6, 5.9, "是", fontsize=8, color=RED)
    ax.text(6, 1.0, "再结合：可解释性要求、训练/推理成本、样本规模做权衡",
            ha="center", fontsize=10, color=GREY,
            bbox=dict(boxstyle="round", fc="#FEF9E7", ec=ORANGE))
    ax.set_title("图19  模型选型决策流程图", fontsize=14, fontweight="bold")
    save(fig, "fig19_model_selection.png")


# ---------------------------------------------------------------------------
# 20. 时间序列交叉验证（滚动/扩展窗口）
# ---------------------------------------------------------------------------
def fig20():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n = 20
    # expanding window
    ax = axes[0]
    for k in range(4):
        train = k * 0 + np.arange(0, 6 + k * 3)
        test = np.arange(6 + k * 3, 9 + k * 3)
        ax.barh(k, len(train), color=BLUE, edgecolor="w")
        ax.barh(k, len(test), left=len(train), color=ORANGE, edgecolor="w")
    ax.set_title("扩展窗口 (Expanding)")
    ax.set_yticks(range(4)); ax.set_yticklabels([f"Fold {i+1}" for i in range(4)])
    ax.set_xlabel("时间 →"); ax.invert_yaxis()
    # rolling window
    ax2 = axes[1]
    for k in range(4):
        start = k * 3
        ax2.barh(k, 6, left=start, color=BLUE, edgecolor="w")
        ax2.barh(k, 3, left=start + 6, color=ORANGE, edgecolor="w")
    ax2.set_title("滚动窗口 (Rolling)")
    ax2.set_yticks(range(4)); ax2.set_yticklabels([f"Fold {i+1}" for i in range(4)])
    ax2.set_xlabel("时间 →"); ax2.invert_yaxis()
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="训练集"), Patch(color=ORANGE, label="验证/测试集")],
               loc="lower center", ncol=2)
    fig.suptitle("图20  时间序列交叉验证（训练集始终早于验证集）", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    save(fig, "fig20_ts_cv.png")


if __name__ == "__main__":
    for f in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10,
              fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19, fig20]:
        f()
    print("ALL DONE")

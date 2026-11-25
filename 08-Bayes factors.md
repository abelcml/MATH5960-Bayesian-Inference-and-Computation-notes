# Week 8 — Bayes factors（原 repo 已整理部分）

## 1. 我們在比較什麼？

**Parameter inference：** 固定模型，推未知 $\theta$。

**Model comparison：** 比較模型 $M_0$、$M_1$，問資料讓哪個模型的支持度增加。

## 2. Marginal likelihood

每個模型先把自己的 parameters 積分掉：

$$
\boxed{m_i(x)=p(x\mid M_i)=\int L(x\mid\theta_i,M_i)\pi(\theta_i\mid M_i)\,d\theta_i}
$$

**意思：** 模型 $M_i$ 在考慮整個 prior parameter uncertainty 後，對這份資料的 predictive density。

## 3. Bayes factor = 兩個 marginal likelihood 相比

$$
\boxed{B_{01}=\frac{m_0(x)}{m_1(x)}}
$$

$$
\boxed{\frac{\Pr(M_0\mid x)}{\Pr(M_1\mid x)}
=B_{01}\,\frac{\Pr(M_0)}{\Pr(M_1)}}
$$

**背法：posterior odds = Bayes factor × prior odds。**

- $B_{01}>1$：資料把 odds 推向 $M_0$，**不表示** $\Pr(M_0\mid x)=B_{01}$。
- 通常需要 **proper parameter priors**，因為 improper prior 的任意比例常數會污染 marginal likelihood。

## 本 repo 尚未整理

原版的「Bayesian hypothesis testing」圖片連結缺少實際圖片檔；「Bayesian model comparison」仍是空章。本次不憑空補上其證明或例題。

## 考試要會

定義 $m_i(x)$、$B_{01}$，以及 posterior odds 與 prior odds 的關係。

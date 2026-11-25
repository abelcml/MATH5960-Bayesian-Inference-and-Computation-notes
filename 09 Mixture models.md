# Week 9 — Mixture models（原 repo 已整理部分）

## 1. Mixture model 在做什麼？

**一句話：** 每筆資料可能來自不同群（component），但我們不知道它真正來自哪群。

若有 $M$ 個 components：

$$
\boxed{f(x_i\mid\theta,\lambda)=\sum_{m=1}^M\lambda_m f_m(x_i\mid\theta_m)}
$$

- $\lambda_m$：落在第 $m$ 群的機率，$\lambda_m\ge0$、$\sum_m\lambda_m=1$。
- $\theta_m$：第 $m$ 群自己的參數。
- $f_m$：第 $m$ 群的 density。

## 2. Latent allocation $z_i$

**$z_i$ 是隱藏的 group label**。定義 $z_{im}=1$ 表示第 $i$ 筆來自群 $m$，否則 0；每筆只有一個 1。

$$
\Pr(z_{im}=1\mid\lambda)=\lambda_m,\quad
\sum_{m=1}^M z_{im}=1
$$

$$
\boxed{p(z_i\mid\lambda)=\prod_{m=1}^M\lambda_m^{z_{im}}}
$$

直覺：**沒看到標籤時是 mixture sum；補上隱藏標籤後，可以按 component 拆開。**

## 3. 本 repo 尚未整理

原版只有 Eyes data 的 mixture / latent allocation；「Change-point models」與「Reaction times and schizophrenia」是空標題。本次保留範圍，不捏造缺少的內容。

## 考試要會

Mixture likelihood 的加權總和、weights 加總為 1、$z_i$ 是 latent component label。

# MATH5960｜極簡複習筆記

**讀法：先看每章「一句話」，再記公式，最後看「考試要會」。** 只保留原筆記裡已整理的主題；未寫的章節不假裝完成。

## 全 repo 統一 notation（先讀這個）

| 符號 | 固定意思 |
| --- | --- |
| $x=(x_1,\ldots,x_n)$ | 已觀察資料；$n$ 是資料筆數 |
| $\theta$ | 未知參數（可以是一個數或向量） |
| $\pi(\theta)$；$\pi(\theta\mid x)$ | prior；posterior |
| $L(x\mid\theta)$ | likelihood |
| $\tilde\pi(\theta\mid x)$ | 未正規化 posterior；$\pi=\tilde\pi/Z$ |
| $N$；$\theta_t$ | Monte Carlo 次數；MCMC 第 $t$ 個 state |
| $\theta^*$；$\theta_{t+1}$ | 候選 state；接受／拒絕後的新 state |
| $q(\theta^*\mid\theta_t)$ | M-H 的 proposal density：給定目前位置，提出新位置 |
| $g(z)$ | RS / IS 的 easy-to-sample proposal |
| $U\sim U(0,1)$ | 用 uniform random number 實現 accept/reject |
| $h(\theta)$ | 想估計 expectation 的函數 |
| $v$ | **若用於 Normal model**，代表 variance $\sigma^2$，不是 standard deviation |

**寫法規則：** $N(m,v)$ 的第二參數是 **variance**；$\operatorname{Gamma}(a,b)$ 用 **shape–rate**；$\operatorname{InvGamma}(a,b)$ kernel 為 $v^{-a-1}e^{-b/v}$。不同章的 $\theta$ 一律代表未知參數，不要一會用 $X_t$ 一會用 $\theta^{(t)}$ 表示同一條 MCMC chain。

## 整門課一句話地圖

1. [01 Basics](01-basics.md)：Bayes 更新；credible interval；predictive。
2. [02 Priors & inversion](02-priors-and-inversion-sampling.md)：conjugate / improper / Jeffreys；inverse CDF。
3. [03 Multiparameter, MC & RS](03-multiparameter-mc-rejection.md)：算積分難 → 用樣本平均；沒法抽樣 → RS。
4. [04 Decisions, asymptotics & IS](04-decisions-asymptotics-mc-error.md)：posterior 如何做決策、近似與加權估計。
5. [05 MCMC & Gibbs](05-mcmc.md)：靠 Markov chain 間接取得 posterior samples。
6. [07 M-H](07-MH.md)：propose → accept/reject；Gibbs 是特例。
7. [08 Bayes factors](08-Bayes%20factors.md)：比較模型的 marginal likelihood。
8. [09 Mixture models](09%20Mixture%20models.md)：隱藏 component labels、mixture likelihood。

目前 repo 沒有 Week 6、Week 10 的筆記檔；Week 8、9 原版部分章節未整理。

## 遇到 sampler 題目先問三句

1. **目標 $\pi$ 是什麼？** 我們要從哪個 distribution 抽？
2. **候選點怎麼來？** 直接抽、RS、IS 的 $g$，還是 M-H 的 $q$？
3. **如何保證正確？** RS 接受後為 target；IS 靠 weights 算積分；M-H 靠 detailed balance；Gibbs 直接從 full conditional 抽。

**終點永遠是 inference**：$E_\pi[h(\theta)]\approx N^{-1}\sum_{t=1}^N h(\theta_t)$（MCMC 要檢查收斂與 autocorrelation）。

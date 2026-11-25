## Part 1 Loss functions / Bayesian decisions

**原則：選 decision $d$，讓 posterior expected loss 最小。**

$$
\boxed{d^*=\arg\min_d E_{\pi(\theta\mid x)}[L(\theta,d)]}
$$

| Loss | 最佳 decision |
| --- | --- |
| Squared error $(\theta-d)^2$ | Posterior **mean** |
| Absolute error $|\theta-d|$ | Posterior **median** |
| Asymmetric linear：低估成本 $k_1$、高估成本 $k_2$ | Posterior quantile $k_1/(k_1+k_2)$ |
| 0–1 loss（離散 $\theta$） | Posterior **mode** |

Continuous $\theta$ 的 exact 0–1 loss 不適合直接用來推 mode；這個結果限定離散狀況。

---

## Part 2 Posterior asymptotics

**一句話：資料多時，在 regularity conditions 下 posterior 接近 Normal，且越來越窄。**

$$
\boxed{\theta\mid x\approx N\left(\hat\theta,\frac1{nI(\theta_0)}\right)}
$$

- $\hat\theta$：MLE，中心。
- $I(\theta_0)$：每筆 observation 的 Fisher information。
- Variance 約 $1/n$；SD 約 $1/\sqrt n$。

**為什麼？** 在 MLE 附近，log-likelihood Taylor expansion 的一階導數 $=0$，剩下負的 quadratic term，形狀就是 Normal kernel：

$$
\log L(x\mid\theta)\approx\log L(x\mid\hat\theta)
-\tfrac12 I_n(\hat\theta)(\theta-\hat\theta)^2
$$

這裡 $I_n(\hat\theta)=-\partial^2\log L/\partial\theta^2|_{\hat\theta}$ 是整份資料的 observed information，且 $I_n\approx nI(\theta_0)$。

**限制：** regular、identifiable、interior true parameter、適當 prior；不是所有模型都成立。

---

## Part 3 Monte Carlo error

對 **i.i.d.** posterior samples $\theta_1,\ldots,\theta_N$：

$$
\hat\mu=\frac1N\sum_{j=1}^N h(\theta_j),\qquad
\boxed{\operatorname{MCSE}\approx s_h/\sqrt N}
$$

要把誤差約減半，$N$ 需變 **4 倍**。

MCMC samples 有 autocorrelation；**不能直接把 i.i.d. 的 MCSE 公式照搬**，要考慮 long-run variance / ESS。

---

## Part 4 Importance sampling

**問題：** target $p(z)$ 不好抽，但想求 $E_p[h(Z)]$。

選容易抽的 proposal $g(z)$（要覆蓋 target 的 support），抽 $Z_j\overset{\mathrm{iid}}{\sim}g$，給每筆 weight：

$$
w_j=\frac{p(Z_j)}{g(Z_j)}
$$

若 $p$ 已 normalised：

$$
\boxed{\hat\mu_{\rm IS}=\frac1N\sum_{j=1}^N h(Z_j)w_j}
$$

若 target 只知 $\tilde p(z)$（未知 normalising constant），改用 **self-normalised IS (SNIS)**：

$$
\tilde w_j=\frac{\tilde p(Z_j)}{g(Z_j)},\quad
W_j=\frac{\tilde w_j}{\sum_k\tilde w_k},\quad
\boxed{\hat\mu_{\rm SN}=\sum_j W_j h(Z_j)}
$$

**記：普通 IS 前面有 $1/N$；SNIS 的 $W$ 已加總成 1，不再乘 $1/N$。**

- Ordinary IS（適當條件）unbiased。
- SNIS 一般 finite-sample biased，但 consistent。
- 少數 weights 極大 → estimator 不穩；proposal 需要覆蓋 target 的重要區域與尾部。

### IS 跟 RS 差在哪？

- **IS：** 不丟 proposal，給 weights，目的主要是 **算積分**。
- **RS：** 不合格就丟掉重抽，accepted samples **真的來自 target**。
- 因此原始 weighted IS samples **不能直接當 Gibbs 的 exact conditional draw**。

**考試要會**

Loss → mean / median / quantile / mode；大樣本 Normal approximation；MCSE 的 $1/\sqrt N$；IS / SNIS 公式與差別。

## 1. Usual distributions

| Kernel（忽略與參數無關的常數） | 分布 |
| --- | --- |
| $e^{-(\theta-m)^2/(2v)}$ | $N(m,v)$ |
| $\theta^{a-1}e^{-b\theta}$，$\theta>0$ | $\operatorname{Gamma}(a,b)$ |
| $v^{-a-1}e^{-b/v}$，$v>0$ | $\operatorname{InvGamma}(a,b)$ |
| $\theta^{a-1}(1-\theta)^{b-1}$，$0<\theta<1$ | $\operatorname{Beta}(a,b)$ |
| $(1+t^2/\nu)^{-(\nu+1)/2}$ | Student $t_\nu$ |

約定：Normal 寫 **mean, variance**；Gamma 寫 **shape, rate**。

---

## 2. Bayesian updating

**資料用 likelihood，信念用 prior，更新後得到 posterior；最後用 posterior 做 inference / prediction。**

$$
\boxed{\pi(\theta\mid x)\propto L(x\mid\theta)\pi(\theta)}
$$

完整式：$\pi(\theta\mid x)=L(x\mid\theta)\pi(\theta)/\int L(x\mid u)\pi(u)\,du$。

### Posterior predictive sampling

$$
\boxed{p(y\mid x)=\int f(y\mid\theta)\pi(\theta\mid x)\,d\theta}
$$

如果已經有 posterior samples：先抽 $\theta$，再抽 $y\mid\theta$。**先抽參數，再抽新資料**，即把 parameter uncertainty 一起帶入預測。

---

## 3. Credible intervals

想找 posterior 裡包含 $\theta$ 的 95% 機率區間。

- **Equal-tailed**：直接取 posterior 2.5% 和 97.5% quantiles。
- **HPD / HDR**：優先收進 posterior density 較高的位置；multimodal 時可能不連續。

$$
\Pr(a\le\theta\le b\mid x)=0.95
$$

這是 **給定資料後，參數的 posterior probability**，不是 frequentist confidence interval 的解釋。

**考試要會**

寫出 Bayes formula、從 kernel 認出 distribution、解釋 credible interval、寫 posterior predictive 的兩段抽樣。

# 04 Bayesian Decisions, Posterior Asymptotics, Monte Carlo Error, Importance Sampling

## Part 1 Loss functions / Bayesian decisions

選 decision $d$ 使 expected posterior loss $E_\pi[L(\theta,d)]=\int L(\theta,d)\,\pi(\theta\mid x)\,d\theta$ 最小。

| Loss | $L(\theta,d)$ | 最佳 decision $d$ |
| --- | --- | --- |
| Quadratic | $(\theta-d)^2$ | posterior mean |
| Linear | $k_1(\theta-d)$ if $d<\theta$；$k_2(d-\theta)$ if $d>\theta$ | posterior quantile $q=\frac{k_1}{k_1+k_2}$ |
| Absolute error | $\lvert\theta-d\rvert$ | posterior median |
| 0-1 | 猜錯就是 1 | posterior mode |

- For quadratic loss, the minimum expected loss is the posterior variance：
$$
E_\pi[(\theta-d)^2]=\text{Var}(\theta\mid x)+\big(E(\theta\mid x)-d\big)^2
$$
- For linear loss, the minimum when takes decision $d$ as quantile $q$.
- Absolute error loss $L(\theta,d)=\lvert\theta-d\rvert$ is a simplified version of linear loss ($k_1=k_2$). Min happens when we pick $d$ as median.
- 0-1 loss：pick mode（posterior density 最高的點）。

---

## Part 2 Posterior asymptotics

### 2.1 結論

- Freq：MLE 接近真實參數，當 $n$ 越來越大（估計越來越準）
- Bayes：當樣本數 $n\to\infty$，整個 posterior distribution 會越來越集中在真實參數附近（對參數的不確定性越來越小）

Asymptotic normal：

$$
\pi(\theta\mid x)\to N\big(\theta_0,\;I_n(\theta_0)^{-1}\big)
$$

And we can use frequentist MLE $\hat\theta$ 來充當 $\theta_0$：

$$
\theta\mid x\approx N\big(\hat\theta,\;I_n(\hat\theta)^{-1}\big),\qquad I_n(\theta)=-\frac{\partial^2\log L(x\mid\theta)}{\partial\theta^2}
$$

### 2.2 大樣本下用期望來代替

令 $\ell_n(\theta)=\frac1n\sum_{i=1}^n\log f(x_i\mid\theta)$，則

$$
\pi(\theta\mid x)\propto\pi(\theta)\exp\big(n\,\ell_n(\theta)\big)
$$

$\ell_n(\theta)$ 依賴隨機抽到的資料，很難直接分析。所以在大樣本下，改成用期望值來近似。

By LLN we have

$$
\ell_n(\theta)\approx E[\log f(x\mid\theta)]
$$

Also

$$
E[\ell_n(\theta)]=E\left[\frac1n\sum_{i=1}^n\log f(x_i\mid\theta)\right]=\frac1n\times nE[\log f(x\mid\theta)]=E[\log f(x\mid\theta)],\quad\forall n
$$

These two are same, so we can let

$$
\pi(\theta\mid x_1,\ldots,x_n)\;\underset{\sim}{\propto}\;\pi(\theta)\exp\big(nE[\ell_n(\theta)]\big)
$$

### 2.3 真實參數的 posterior density，相對於錯誤參數，到底高多少？

我們發現隨著資料增加，真實參數相對於錯誤參數，會獲得越來越強的 posterior 支持：

$$
\lim_{n\to\infty}\frac{\pi(\theta_0\mid x)}{\pi(\theta\mid x)}=\infty
$$

### 2.4 Posterior 集中在 $\theta_0$ 附近時，它的形狀是什麼？

在 $\theta_0$ 附近用 Taylor 展開（一階項為 0，因為 $\theta_0$ 是最大值點）：

$$
E[\ell_n(\theta)]\approx E[\ell_n(\theta_0)]-\frac12 I(\theta_0)(\theta-\theta_0)^2
$$

其中 $I(\theta_0)=-E\left[\frac{\partial^2\log f(x\mid\theta)}{\partial\theta^2}\right]_{\theta_0}$ 是單一觀測的 Fisher information。

把近似放回 posterior：

$$
\pi(\theta\mid x)\propto\pi(\theta)\exp\big(nE[\ell_n(\theta)]\big)
$$

當 $n$ 很大、prior 在 $\theta_0$ 附近平滑且非零時，prior 可以近似看成常數。所以：

$$
\pi(\theta\mid x)\;\underset{\sim}{\propto}\;\exp\left[n\left(C-\frac{(\theta-\theta_0)^2}{2v}\right)\right]\propto\exp\left[-\frac{n(\theta-\theta_0)^2}{2v}\right]
$$

可以發現 $v=I(\theta_0)^{-1}$，以及

$$
\theta\mid x\approx N\left(\theta_0,\;\frac vn\right)
$$

So posterior variance is $I(\theta_0)^{-1}/n=I_n(\theta_0)^{-1}$。

---

## Part 3 Monte Carlo error

Recall Monte Carlo integration 就是把 integration 轉換成 expectation，再用 sample average 估計：

$$
\bar g=\frac1N\sum_{i=1}^N g(\theta^{(i)})\approx E_\pi[g(\theta)]
$$

By CLT：

$$
\bar g\approx N\left(E_\pi[g],\;\frac{s_g^2}{N}\right),\qquad s_g^2=\frac1N\sum_i\big(g(\theta^{(i)})-\bar g\big)^2
$$

- Monte Carlo standard error $=s_g/\sqrt N$，誤差以 $1/\sqrt N$ 的速度變小
- 要讓 $\bar g$ 以機率 $1-\alpha$ 誤差在 $\varepsilon$ 以內：

$$
z_{1-\alpha/2}\frac{s_g}{\sqrt N}\le\varepsilon\quad\Rightarrow\quad N\ge\left(\frac{z_{1-\alpha/2}\,s_g}{\varepsilon}\right)^2
$$

---

## Part 4 Importance sampling

Target $p$ 不好抽，改從容易抽的 $q$ 抽，再用 weight 修正：

$$
E_p[h(x)]=\int h(x)\frac{p(x)}{q(x)}q(x)\,dx\approx\frac1N\sum_{i=1}^N h(x_i)\,w(x_i),\qquad x_i\sim q,\quad w=\frac pq
$$

$p$ 只知道 up to constant 時，用 normalised weights：

$$
E_p[h]\approx\frac{\sum_i h(x_i)\,w(x_i)}{\sum_i w(x_i)}
$$

- 不用丟樣本（跟 rejection sampling 不同）
- $q$ 的 tail 要比 $p$ 重，不然 weight 會爆掉、variance 很大

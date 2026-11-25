## Part 1 Markov chains / MCMC

### 5. Page 17: Why MCMC

**我們真正要做 Bayesian inference**：算 $E_{\pi(\theta\mid x)}[h(\theta)]$。但 posterior 太難直接抽，所以設計一條 **stationary distribution = posterior** 的 Markov chain，拿它的長期 samples 算平均。

$$
\boxed{\frac1N\sum_{t=1}^N h(\theta_t)\longrightarrow E_\pi[h(\theta)]}
$$

MCMC samples 通常相依，**不是 i.i.d.**。

### 1. Definition

- $\theta_t$：現在 state；$\theta_{t+1}$：下一個 state。
- **Markov property**：下一步只需要知道現在，不需知道更早歷史。
- $P(a,b)=\Pr(\theta_{t+1}=b\mid\theta_t=a)$（離散 state）。
- $P^k(a,b)$：走 $k$ 步後到 $b$ 的機率。
- **Stationary**：$\pi P=\pi$，若現在分布是 $\pi$，走一步仍是 $\pi$。

兩步轉移要加總所有中途位置：

$$
P^2(a,b)=\sum_c P(a,c)P(c,b)
$$

### 3. Irreducibility and aperiodicity

| 名稱 | 人話 |
| --- | --- |
| **Irreducible** | 想去的區域都到得了，不會卡死在某一區。 |
| **Aperiodic** | 不被固定週期綁住。 |
| **Positive recurrent** | 有適當的回返性，使 stationary probability 存在。 |
| **Ergodic theorem** | 跑夠久，time average → target expectation。 |

對離散 irreducible chain，state $a$ 的 period：

$$
d(a)=\gcd\{k\ge1:P^k(a,a)>0\}
$$

$d(a)=1$ → aperiodic。**注意：** time-average convergence 本身通常不需要 aperiodicity；分布收斂到 stationary 則還需相關條件。連續 state 需要相應的 Harris 類條件。

---

## Part 2 Gibbs sampling

### 1. Update full conditionals

假設 $\theta_t=(a_t,b_t)$，目標是 joint posterior $\pi(a,b\mid x)$。

**一輪 Gibbs：**

1. 固定 $b_t$，抽 $a^*\sim\pi(a\mid b_t,x)$。
2. 固定剛抽的 $a^*$，抽 $b^*\sim\pi(b\mid a^*,x)$。
3. 記錄 **$\theta_{t+1}=(a^*,b^*)$**。

**Full conditional** = 抽某個參數時，固定 **其餘全部參數** 與資料。

**為何用 Gibbs？** Joint posterior 難直接抽，但每個 full conditional 可能很好抽。每一步要 **exact conditional sampling**。

### 2. Example P25: Normal model

若 $x_i\mid\mu,v\sim N(\mu,v)$，prior 獨立為 $\mu\sim N(\xi,\kappa^{-1})$ 和 $1/v\sim\operatorname{Gamma}(\alpha,\beta)$：

$$
\mu\mid v,x\sim N\left(
\frac{\sum_i x_i/v+\kappa\xi}{n/v+\kappa},\;
\frac1{n/v+\kappa}\right)
$$

$$
(1/v)\mid\mu,x\sim\operatorname{Gamma}\left(
\alpha+n/2,\ \beta+\tfrac12\sum_i(x_i-\mu)^2
\right)
$$

先用目前 $v$ 抽 $\mu$，再用新的 $\mu$ 抽 $1/v$。

### 3. P27–31: Failure of Gibbs

- **無法從 full conditional 直接抽** → Week 7 的 M-H。
- **Parameters 高度相關** → coordinate-wise moves 很小，mixing 慢、autocorrelation 高。
- **Irreducibility 失敗** → 可能永遠走不到某些區域。

改善：**blocking**（高度相關的參數一起抽）；**reparameterisation**（換座標讓 dependency 變小）。

---

## Part 3 Assessing convergence

| 方法 | 目的 |
| --- | --- |
| Burn-in | 丟掉可能受起點影響的前段；不是丟了就保證收斂。 |
| Trace plot | 看 chain 有沒有停住、漂移或混合不好。 |
| Multiple chains + $\hat R$ | 不同起點的 chains 是否看起來一致；$\hat R\approx1$ 非保證。 |
| ACF / ESS | 衡量 autocorrelation、有效資訊量。 |
| MCSE | 衡量 estimator 的 Monte Carlo 不確定性。 |
| Thinning | 每隔幾步存一筆，主要節省儲存；通常不是提升效率的首選。 |

**考試要會：** Markov property / stationary / irreducible / aperiodic；Gibbs 的 full conditional 更新順序；burn-in、mixing、blocking 的作用。

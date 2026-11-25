## Part 1 Loss functions / Bayesian decisions

選 decision $d$ 使 expected posterior loss $E_\pi[L(\theta,d)]=\int L(\theta,d)\,\pi(\theta\mid x)\,d\theta$ 最小。

| Loss | $L(\theta,d)$ | 最佳 decision $d$ |
| --- | --- | --- |
| Quadratic | $(\theta-d)^2$ | posterior mean |
| Linear | $k_1(\theta-d)$ if $d<\theta$；$k_2(d-\theta)$ if $d>\theta$ | posterior quantile $q=\frac{k_1}{k_1+k_2}$ |
| Absolute error | $\lvert\theta-d\rvert$ | posterior median |
| 0-1（discrete parameter） | $\mathbf1\{\theta\ne d\}$ | posterior mode |

- For quadratic loss, the minimum expected loss is the posterior variance：
$$
E_\pi[(\theta-d)^2]=\text{Var}(\theta\mid x)+\big(E(\theta\mid x)-d\big)^2
$$
- For linear loss, the minimum when takes decision $d$ as quantile $q$.
- Absolute error loss $L(\theta,d)=\lvert\theta-d\rvert$ is a simplified version of linear loss ($k_1=k_2$). Min happens when we pick $d$ as median.
- Discrete 0-1 loss：pick mode（posterior probability 最大的點）。Continuous posterior 下，exact 0-1 loss 對每個 $d$ 的 expected loss 都是 1；若改用 $L_\varepsilon(\theta,d)=\mathbf1\{|\theta-d|>\varepsilon\}$，小 $\varepsilon$ 下才會在適當條件下連到 density mode。

---

## Part 2 Posterior asymptotics

### 1. 樣本變多，posterior 會怎樣？

資料越多，對 $\theta$ 的不確定性通常越小。在 regular model 下，posterior 會集中到真實參數 $\theta_0$ 附近，而且局部形狀越來越接近 normal：

$$
\theta\mid x\approx N\left(\hat\theta,\;\frac{1}{nI(\theta_0)}\right)
$$

這裡 $\hat\theta$ 是 MLE，$I(\theta_0)$ 是 **一筆 observation 的 Fisher information**。先看懂兩件事：

- **位置**：posterior 的中心約在 $\hat\theta$，而 $\hat\theta\to\theta_0$。
- **寬度**：normal approximation 的 variance 是 $1/[nI(\theta_0)]$，standard deviation 以 $1/\sqrt n$ 的速度縮小。

以下考慮 scalar parameter、i.i.d. data 和 correctly specified model。這個結果需要 regularity conditions，例如 model identifiable、$\theta_0$ 在參數空間內部、log-likelihood 足夠平滑、$0<I(\theta_0)<\infty$，且固定的 prior 在 $\theta_0$ 附近連續且正值。下面是推導直覺。

---

### 2. 從 posterior 開始，在 MLE 附近做 Taylor

保留原本的記號，$\ell_n(\theta)$ 是 **平均** log-likelihood：

$$
\ell_n(\theta)=\frac1n\sum_{i=1}^n\log f(x_i\mid\theta),
\qquad
\pi(\theta\mid x)\propto\pi(\theta)\exp\big(n\ell_n(\theta)\big)
$$

MLE $\hat\theta$ 是 log-likelihood 的最大值點。若它在內部，則 $\ell_n'(\hat\theta)=0$，所以 Taylor 展開的一階項消失：

$$
\ell_n(\theta)\approx\ell_n(\hat\theta)
+\frac12\ell_n''(\hat\theta)(\theta-\hat\theta)^2
$$

定義 **整份資料的 observed information**：

$$
I_n(\hat\theta)
=-\frac{\partial^2\log L(x\mid\theta)}{\partial\theta^2}\bigg|_{\theta=\hat\theta}
=-n\ell_n''(\hat\theta)
$$

在 peak 附近，log-likelihood 向下彎，所以 $I_n(\hat\theta)>0$。乘回 $n$，上面的 Taylor 就變成：

$$
n\ell_n(\theta)\approx n\ell_n(\hat\theta)
-\frac12 I_n(\hat\theta)(\theta-\hat\theta)^2
$$

曲率 $I_n$ 越大，離開 peak 時 log-likelihood 掉得越快，posterior 就越窄。

---

### 3. 放回 posterior，就看到 normal kernel

把 Taylor approximation 代回去。$\exp(n\ell_n(\hat\theta))$ 跟 $\theta$ 無關，可以收進 normalising constant：

$$
\pi(\theta\mid x)\approx C\,\pi(\theta)
\exp\left[-\frac12 I_n(\hat\theta)(\theta-\hat\theta)^2\right]
$$

當 posterior 集中在很小的範圍，連續且正值的 prior 在這個範圍內變化很小，因而可用 $\pi(\hat\theta)$ 近似。把它也收進 constant：

$$
\pi(\theta\mid x)\approx C'
\exp\left[-\frac12 I_n(\hat\theta)(\theta-\hat\theta)^2\right]
$$

對照 normal density 的 kernel $\exp[-(\theta-\mu)^2/(2\sigma^2)]$，直接讀出：

$$
\boxed{\theta\mid x\approx N\left(\hat\theta,\;I_n(\hat\theta)^{-1}\right)}
$$

所以實際拿到資料後，可以算 MLE 和它附近的曲率，得到 posterior 的 normal approximation。

---

### 4. 為什麼 variance 大約是 $1/n$？

一筆 observation 的 expected Fisher information 是

$$
I(\theta_0)
=-E_{\theta_0}\left[
\frac{\partial^2\log f(X\mid\theta)}{\partial\theta^2}\bigg|_{\theta=\theta_0}
\right]
$$

整份資料的 log-likelihood 是 $n$ 項相加，曲率也會累加。在 regularity conditions 下，LLN 加上 MLE consistency 給出

$$
\frac{I_n(\hat\theta)}{n}\xrightarrow{P}I(\theta_0)
\quad\Rightarrow\quad
I_n(\hat\theta)^{-1}\approx\frac{1}{nI(\theta_0)}
$$

這就接回一開始的結果。$I_n$ 是從這份資料算出的 observed information，$nI(\theta_0)$ 是 expected information；兩者在大樣本下接近。

**Example：** 若 $X_i\sim N(\theta,\sigma^2)$，$\sigma^2$ 已知，則 $\hat\theta=\bar x$、$I_n=n/\sigma^2$。搭配符合上述條件的 prior，大樣本下

$$
\theta\mid x\approx N\left(\bar x,\;\frac{\sigma^2}{n}\right)
$$

資料量變成 4 倍，posterior standard deviation 約縮成一半。

**$\hat\theta$ 和 $\theta_0$ 要分清楚：** 重抽一份資料，$\hat\theta$ 就會移動；posterior 描述的是給定目前資料後，$\theta$ 的不確定性。雖然 $\hat\theta\to\theta_0$，兩者的距離通常仍與 posterior 寬度同為 $1/\sqrt n$ 的量級，所以 normal approximation 要保留以 $\hat\theta$ 為中心。

---

## Part 3 Monte Carlo error

Recall Monte Carlo integration 就是把 integration 轉換成 expectation，再用 sample average 估計：

$$
\bar g=\frac1N\sum_{i=1}^N g(\theta^{(i)})\approx E_\pi[g(\theta)]
$$

假設 $\theta^{(i)}$ 是 i.i.d. posterior draws，且 $g(\theta)$ 有有限 variance。By CLT，大 $N$ 時：

$$
\bar g\approx N\left(E_\pi[g],\;\frac{s_g^2}{N}\right),\qquad s_g^2=\frac1{N-1}\sum_{i=1}^N\big(g(\theta^{(i)})-\bar g\big)^2
$$

- Monte Carlo standard error $=s_g/\sqrt N$，誤差以 $1/\sqrt N$ 的速度變小
- 用 normal approximation 規劃誤差在 $\varepsilon$ 內、coverage 約 $1-\alpha$ 的 sample size（$s_g$ 可先由 pilot sample 估計）：

$$
z_{1-\alpha/2}\frac{s_g}{\sqrt N}\le\varepsilon\quad\Rightarrow\quad N\ge\left(\frac{z_{1-\alpha/2}\,s_g}{\varepsilon}\right)^2
$$

MCMC samples 有 autocorrelation，需要另外估計 long-run variance / effective sample size，不能直接套用 i.i.d. 的 $s_g/\sqrt N$。

---

## Part 4 Importance sampling

想估 $\mu=E_p[h(X)]$，但 target $p$ 不好抽時，改從容易抽的 proposal $q$ 抽樣，再給每筆 sample weight。$q(x)>0$ 需涵蓋 $p(x)>0$ 的區域。

### 1. Target 已 normalised

1. 抽 i.i.d. $X_i\sim q$。
2. 算 weight $w_i=p(X_i)/q(X_i)$。
3. 用 weighted average 估計：

$$
\mu=\int h(x)\frac{p(x)}{q(x)}q(x)\,dx,
\qquad
\hat\mu=\frac1N\sum_{i=1}^N h(X_i)w_i
$$

若 $E_p[|h(X)|]<\infty$，則 $E_q[\hat\mu]=\mu$，ordinary importance sampling 是 unbiased。

### 2. Target 只知道 up to constant

若 $p(x)=\tilde p(x)/Z$，但 $Z$ 不知道，先算 unnormalised weights：

$$
\tilde w_i=\frac{\tilde p(X_i)}{q(X_i)},
\qquad
W_i=\frac{\tilde w_i}{\sum_{j=1}^N\tilde w_j}
$$

Self-normalised estimator：

$$
\hat\mu_{\mathrm{SN}}
=\sum_{i=1}^N W_i h(X_i)
=\frac{\sum_i\tilde w_i h(X_i)}{\sum_i\tilde w_i}
$$

$W_i$ 已經加總為 1，外面不用再乘 $1/N$。這個 ratio estimator 一般有 finite-sample bias；在 support 與 integrability 條件成立時 consistent。

### 3. Proposal 要怎麼選

保留所有 samples，以 weights 調整貢獻。若少數 weights 特別大，estimate 會不穩。選擇涵蓋 target 重要區域、tail 不太輕的 $q$ 通常較安全，但仍需看 $h$ 與 weight 的 second moment。

Ordinary IS 的 finite-variance 條件是

$$
E_q[(h(X)w(X))^2]
=\int\frac{h(x)^2p(x)^2}{q(x)}\,dx<\infty
$$

## Part 1 Mixture priors, prior predictive checking

### 1.1 Mixture priors

單一 Beta distribution 不夠靈活，因此我們可以混合兩個 Beta distributions：

$$
\pi(\theta)=w_1\pi_1(\theta)+w_2\pi_2(\theta)
$$

其中 $w_1,w_2\ge0$ 且 $w_1+w_2=1$。

例如：

$$
\pi(\theta)=0.5\,\text{Beta}(40,60)+0.5\,\text{Beta}(60,40)
$$

### 1.2 Prior predictive checking

Prior predictive distribution：

$$
p(x)=\int f(x\mid\theta)\,\pi(\theta)\,d\theta
$$

做法：從 prior 抽 $\theta$，再從 likelihood 抽 $x$，看模擬出來的資料合不合理。不合理時，要回頭檢查 prior 和 likelihood 的設定。

---

## Part 2 Multi-parameter Bayesian inference

### 2.1 從 scalar 到 vector

Unknown quantity from scalar to vector：$\boldsymbol{\theta}=(\theta_1,\theta_2,\ldots,\theta_d)$。

### 2.2 變的地方

1. Prior 是 $\pi(\theta_1,\theta_2)$，posterior 是 $\pi(\theta_1,\theta_2\mid x)$。兩個 continuous parameters 時，可以畫成 density surface：
   - x-axis = $\theta_1$
   - y-axis = $\theta_2$
   - height = posterior density
2. Prior specification 必須描述 dependency between $\theta_1,\theta_2$。
3. Interpretation：joint posterior 的 $\text{Cov}(\theta_1,\theta_2)$ 能告訴額外資訊。例如 $\theta_1$ 本身不確定，$\theta_2$ 本身不確定，但 $\theta_1+\theta_2$ 或 $\theta_1-\theta_2$ 幾乎等於某個值。

### 2.3 步驟（以 Normal 為例）

1. 選 reference prior（也常稱 independence Jeffreys prior）$\pi(\mu,\sigma^2)\propto1/\sigma^2$，density 是相對於 $d\mu\,d\sigma^2$
2. 乘 likelihood，算 posterior
3. Algebraic trick：$\sum_i(y_i-\mu)^2=(n-1)s^2+n(\bar y-\mu)^2$
4. Integrate out 不要的參數

注意：直接對 $(\mu,\sigma^2)$ 用 joint Jeffreys rule，會得到 $(\sigma^2)^{-3/2}$。以下結果全部使用上面的 $1/\sigma^2$ prior，假設 $n\ge2$ 且 $s^2>0$。

### 2.4 Conclusion

$$
p(\mu\mid\sigma^2,y)\propto\exp\left(-\frac{n(\bar y-\mu)^2}{2\sigma^2}\right)
\quad\Rightarrow\quad
\mu\mid\sigma^2,y\sim N(\bar y,\sigma^2/n)
$$

- $\sigma^2\mid y\sim\text{InvGamma}\left(\frac{n-1}{2},\frac{(n-1)s^2}{2}\right)$，或 inverse $\chi^2$
- $\mu\mid y$：substitution 得到 $\dfrac{\mu-\bar y}{s/\sqrt n}\sim t_{n-1}$

### 2.5 怎麼從 joint posterior 抽樣

高維 joint posterior 的 sampling 很難。舉例 Normal：

Recall $p(a,b\mid y)=p(a\mid y)p(b\mid a,y)$，令 $a=\sigma^2$、$b=\mu$，hence

$$
p(\mu,\sigma^2\mid y)=p(\sigma^2\mid y)\cdot p(\mu\mid\sigma^2,y)
$$

可以先抽 $\sigma^2$ 的 marginal posterior，再抽 $\mu$ 的 conditional posterior。

- Joint posterior 可寫成 marginal × conditional；兩個分布都能抽樣時，就可以 sequentially sample。
- 因為前面已經知道 $\sigma^2\mid y\sim\text{InvGamma}$ / Inv-$\chi^2$，而 $\mu\mid\sigma^2,y\sim N(\bar y,\sigma^2/n)$。

### 2.6 Proper conjugate prior

前面使用的 reference prior 是 improper；它仍然代表一種 prior 選擇。

有時候你希望 prior 本身是真正的機率分布。例如：

- 我認為平均數大概在 50 附近。
- 我認為變異數不太可能大於 1000。

那就可以把這些資訊放進 prior，於是出現：**Normal-Inverse-Gamma**

$$
\sigma^2\sim\text{InvGamma}(a,b),\qquad \mu\mid\sigma^2\sim N(m,\sigma^2/\kappa)
$$

### 2.7 Vector version

Recall 我們積分掉 $\mu$ 後，$\sigma^2\mid y\sim\text{InvGamma}\left(\frac{n-1}{2},\frac{(n-1)s^2}{2}\right)$。

Vector version：scalar 的 $\sigma^2$ 用 InvGamma，matrix 的 $\Sigma$ 用 InvWishart。

- $\Sigma\sim\text{InvWishart}$
- precision $\Sigma^{-1}\sim\text{Wishart}$

其 posterior 也是 conjugate：

- If $\Sigma$ known：$\mu\sim N_d\Rightarrow\mu\mid x\sim N_d$
- If $\mu$ known：$\Sigma\sim\text{InvWishart}\Rightarrow\Sigma\mid x\sim\text{InvWishart}$

---

## Part 3 Monte Carlo integration

用「隨機抽樣」來估計積分：

$$
\int h(x)\,p(x)\,dx\approx\frac1N\sum_{i=1}^N h(X_i),\qquad X_i\sim p
$$

LLN 條件：

- 抽樣分布正確
- 抽樣 i.i.d.
- $E_p[|h(X)|]<\infty$

### 3.1 幾種題型

- $[0,1]$ 積分最簡單
- $a$ 到 $b$ 可壓縮到 $[0,1]$
- $0$ 到 $k$ 則設 $u=x/k$
- $0$ to $\infty$ 可設 $u=1/(1+x)$，記得 Jacobian：$x=1/u-1$、$dx=-du/u^2$，所以
  $$
  \int_0^\infty h(x)\,dx=\int_0^1h(1/u-1)\frac{1}{u^2}\,du
  $$

### 3.2 Multidimensional integral

$$
\int_0^1\int_0^1 g(u_1,u_2)\,du_1\,du_2
$$

抽 i.i.d. $U_i=(U_{1i},U_{2i})\sim U([0,1]^2)$，用 $\frac1N\sum_{i=1}^N g(U_{1i},U_{2i})$ 近似這個積分。

### 3.3 Monte Carlo 對 Bayesian computation 的意義

Give me samples from the posterior, and I can estimate almost anything by averaging.

Recall $\int\pi(\mu,\sigma^2\mid y)\,d\sigma^2$ 勉強能算，但當 $\theta\in\mathbb R^{100}$ 就很難。

---

## Part 4 Now how to get samples → Rejection sampling

### 4.1 Why know shape of $f(x)$ but still hard to sample from it

Can calculate density height ≠ can sample from it.

Ex. $f(x)\propto e^{-x^4+x^2}$

General idea is inverse CDF sampling：$U\sim U(0,1)$，then $X=F^{-1}(U)$。

Problem is $F$ or $F^{-1}$ may be hard to calculate.

### 4.2 How rejection sampling works

Condition：$f(x)\ge0$ 且 $0<\int f(x)\,dx<\infty$。只需知道 target up to proportionality。

Step:

1. Find proposal $g(x)$ easy to sample, and constant $K$ s.t. $f(x)\le K g(x)$ for all $x$。
   $g$ 最好接近 $f$ 的形狀，這樣 $K$ 可以小一點（$P(\text{acceptance})\propto 1/K$），acceptance 高一點才不會浪費 proposal。
2. Sample $x^*$ from $g$, and
   - Method 1：draw $y^*$ from $U(0,Kg(x^*))$，then accept $x^*$ if $y^*\le f(x^*)$
   - Method 2：accept with probability $f(x^*)/(Kg(x^*))$

兩個方法相同，因為 $\Pr(y^*\le f(x^*))=f(x^*)/(Kg(x^*))$。

### 4.3 Why accepted samples follow $f$

令 $h(x)=f(x)/(Kg(x))$ 為 $x$ 被接受的機率。

$$
\Pr(X\le x\mid\text{accepted})
=\frac{\int_{-\infty}^x h(y)g(y)\,dy}{\int_{-\infty}^\infty h(y)g(y)\,dy}
=\frac{\int_{-\infty}^x f(y)\,dy}{\int_{-\infty}^\infty f(y)\,dy}
$$

$g$ 和 $K$ 都消掉，剩下的就是 $f$ normalise 之後的 cdf。分母要對整條線積分，因為 $\Pr(\text{accepted})$ 是整體的接受機率。

### 4.4 Bayesian case

$$
\pi(\theta\mid y)\propto L(y\mid\theta)\,\pi(\theta)
$$

RHS can calculate → know relative shape of posterior。

Normalising constant unknown is fine, as long as we can find above envelope.

### 4.5 When to use

Hard to sample target directly, but can calculate $f(x)$ and find easy proposal $g(x)$ with finite $K$ → rejection sampling is one option.

- Normalising constant can be known or unknown, both fine
- 可與 inverse sampling、importance sampling、MCMC 比較成本
- Efficiency depends on how tight envelope is

Let $p(x)=f(x)/Z$，$Z=\int f(x)\,dx$：

$$
\text{acceptance rate}=Z/K
$$

Envelope too loose → reject many candidates → waste samples。要選有限的 $K\ge\sup_{x:g(x)>0}f(x)/g(x)$，並確保 $g>0$ 涵蓋 $f>0$ 的區域。越接近這個上界越有效率；不一定能找到實際的 maximum。

### 4.6 Relation to Monte Carlo integration

Recall use sample average to estimate integral：

$$
\int h(x)\,p(x)\,dx\approx\frac1N\sum_{i=1}^N h(X_i),\qquad X_i\sim p
$$

- Rejection sampling → get samples
- Monte Carlo integration → use samples to estimate integral
- Can sample $p$ directly? Still can do Monte Carlo integration

## 1. Usual distributions

只看跟參數有關的 kernel，constant 可以先忽略。Gamma 用 shape–rate，Inverse-Gamma 用 $v^{-(a+1)}e^{-b/v}$ 的 convention。

| Kernel（對參數而言） | Distribution / support |
| --- | --- |
| $\exp\left(-(\theta-m)^2/(2v)\right)$ | $N(m,v)$，$\theta\in\mathbb R$、$v>0$ |
| $\theta^{a-1}e^{-b\theta}$ | $\operatorname{Gamma}(a,b)$，$\theta>0$ |
| $(\sigma^2)^{-(a+1)}e^{-b/\sigma^2}$ | $\operatorname{InvGamma}(a,b)$，$\sigma^2>0$ |
| $\theta^{a-1}(1-\theta)^{b-1}$ | $\operatorname{Beta}(a,b)$，$0<\theta<1$ |
| $(1+t^2/\nu)^{-(\nu+1)/2}$ | $t_\nu$，$t\in\mathbb R$ |

其中 $a,b,\nu>0$。

---

## 2. Bayesian updating

1. Specify likelihood $L(x\mid\theta)$。
2. Choose prior $\pi(\theta)$。
3. 用 Bayes' theorem 算 posterior：
   $$
   \pi(\theta\mid x)
   =\frac{L(x\mid\theta)\pi(\theta)}
   {\int L(x\mid u)\pi(u)\,du}
   $$
4. 從 posterior 做 inference 或選 decision。

### Posterior predictive sampling

想預測新觀測 $y$，把 parameter uncertainty 積分掉。假設給定 $\theta$ 後，$y$ 與現有資料 $x$ independent：

$$
p(y\mid x)=\int f(y\mid\theta)\pi(\theta\mid x)\,d\theta
$$

已經有 posterior samples 時，可以直接做：

1. 抽 $\theta^{(i)}\sim\pi(\theta\mid x)$。
2. 再抽 $y^{(i)}\sim f(y\mid\theta^{(i)})$。
3. 保留 $y^{(i)}$，就是 posterior predictive samples。

每一對 $(\theta^{(i)},y^{(i)})$ 的 joint density 是 $\pi(\theta\mid x)f(y\mid\theta)$。忽略 $\theta^{(i)}$，就對應到公式中的 integrate out。

---

## 3. Credible intervals

95% credible interval $[a,b]$ 滿足 $\Pr(a\le\theta\le b\mid x)=0.95$。對 continuous posterior，equal-tailed interval 讓兩端各留 2.5%：

$$
\int_{-\infty}^{a}\pi(\theta\mid x)\,d\theta=0.025,
\qquad
\int_{-\infty}^{b}\pi(\theta\mid x)\,d\theta=0.975
$$

積分只需涵蓋 parameter 的 support，例如 $\theta>0$ 時下限可用 $0$。

Highest density region（HDR / HPD region）收集 posterior density 最高的地方：

$$
C_c=\{\theta:\pi(\theta\mid x)\ge c\},
\qquad
\int_{C_c}\pi(\theta\mid x)\,d\theta=0.95
$$

在通常的 continuous density 情況下，這給出最小長度／體積的 credible region。Unimodal posterior 常得到一個 interval；multimodal posterior 的 HDR 可以分成幾段。

不感興趣的 nuisance parameters 可以先 integrate out，再對 marginal posterior 找 credible interval。

# Week 3 — Multiparameter inference, MC & rejection sampling

## 1. Mixture prior / prior predictive

**Mixture prior：** 把不同先驗意見混合。

$$
\pi(\theta)=w_1\pi_1(\theta)+w_2\pi_2(\theta),\quad w_1+w_2=1
$$

**Prior predictive check：** 先抽 $\theta\sim\pi$，再抽 $x\sim f(x\mid\theta)$，看模擬資料是否合理。

$$
p(x)=\int f(x\mid\theta)\pi(\theta)\,d\theta
$$

## 2. 多參數 posterior：joint、marginal、conditional

$\theta=(\theta_1,\theta_2)$ 時，posterior 是 **joint distribution**；不要把兩個參數當作自動獨立。

$$
\pi(\theta_1,\theta_2\mid x)
=\pi(\theta_1\mid\theta_2,x)\pi(\theta_2\mid x)
$$

因此如果右邊兩個分布都好抽：**先抽 $\theta_2\mid x$，再抽 $\theta_1\mid\theta_2,x$**，就得到 joint sample。這叫 sequential sampling，不是 Gibbs。

### Normal model 必記例子

令 $x_i\mid\mu,v\sim N(\mu,v)$，$v=\sigma^2$，且選 $\pi(\mu,v)\propto1/v$（**此處是 independent/reference prior，不是 full Jeffreys**）。定義 $s^2=\sum(x_i-\bar x)^2/(n-1)$，假設 $n\ge2$、$s^2>0$。

$$
\boxed{\mu\mid v,x\sim N(\bar x,v/n)}
$$

$$
\boxed{v\mid x\sim\operatorname{InvGamma}\left(\frac{n-1}{2},\frac{(n-1)s^2}{2}\right)}
$$

$$
\boxed{\frac{\mu-\bar x}{s/\sqrt n}\mid x\sim t_{n-1}}
$$

**抽 joint：** 先抽 $v\mid x$，再抽 $\mu\mid v,x$。最常用的代數技巧：

$$
\sum_{i=1}^n(x_i-\mu)^2=(n-1)s^2+n(\bar x-\mu)^2
$$

Proper conjugate alternative：$v\sim\operatorname{InvGamma}(a,b)$、$\mu\mid v\sim N(m,v/\kappa)$。多維 covariance 可用 inverse-Wishart。

## 3. Monte Carlo integration

**不想硬算積分，改成抽樣後取平均。**

$$
\boxed{E_p[h(X)]=\int h(z)p(z)\,dz
\approx \frac1N\sum_{j=1}^N h(X_j)},\quad X_j\overset{\mathrm{iid}}{\sim}p
$$

- 想估 probability？令 $h$ 為 indicator。
- 想估 mean？令 $h(z)=z$。
- 若積分區間是 $[a,b]$，可以換成 $U(0,1)$；換元要記 Jacobian。
- $\int_0^\infty h(z)\,dz=\int_0^1 h(1/u-1)u^{-2}\,du$。

**最大問題：** Monte Carlo 需要 target samples，但 target 常常很難直接抽。

## 4. Rejection sampling (RS)

**目的：** target $f$ 難抽，但容易從 $g$ 抽，而且能找到 envelope $f(z)\le K g(z)$。

1. 抽候選值 $Z^*\sim g$。
2. 抽 $U\sim U(0,1)$。
3. 若 $U\le f(Z^*)/[K g(Z^*)]$，**接受 $Z^*$**；否則重新抽，直到接受。

這裡 $f$ 可為未正規化 target，$K$ 必須配合它的尺度。

**為何正確？**

$$
p(z\mid\mathrm{accepted})\propto
g(z)\frac{f(z)}{Kg(z)}\propto f(z)
$$

**效率：** 設 $Z=\int f(z)dz$，則 acceptance rate $=Z/K$。$K$ 大 → 拒絕多 → 貴。

## 5. RS vs MC

- **RS 負責生出真正來自 target 的 samples**。
- **MC 負責拿 samples 算 expectations**。
- 能直接抽 target 時，不必特地做 RS。

## 考試要會

分清 joint / marginal / conditional；Normal model 的三個 posterior 結果；MC sample mean；RS 的 envelope、acceptance rule 與 $Z/K$。

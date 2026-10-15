# 02 Priors and Inversion Sampling

## Part 1 Topic of priors

### 1.1 Conjugate prior

Prior that makes the posterior distribution *stay in the same family as the prior*.

Ex.

1. Binomial likelihood, choose $\text{Beta}(a,b)$ to be prior
2. $\text{Gamma}(a,b)$ for Poisson
3. Gamma for Gamma
4. Normal prior $N(b,d^2)$ for Normal likelihood

When data comes from exponential family:

$$
f(x\mid\theta)=h(x)\,g(\theta)\exp\{t(x)\,c(\theta)\}
$$

then the conjugate prior is easy to be found:

$$
\pi(\theta)\propto g(\theta)^{d}\exp\{b\,c(\theta)\}
$$

For normal case, posterior precision = prior precision ($1/d^2$) + $n\times$ data precision ($1/\sigma^2$).

### 1.2 Improper priors

Prior 積分（或求和）是 $\infty$，無法 normalise。

判斷方法：看 support（參數可能的取值範圍）

- support 有界，例如 $p\in[0,1]$ → uniform prior 是 proper
- support 無界，例如 $n\in\{0,1,2,\dots\}$、$\mu\in\mathbb R$ → uniform prior 是 improper

Improper prior 可以用，但 posterior 必須是 proper。

### 1.3 Jeffreys priors

情景：假設你說你對身高 $h$ ignorant，so you set $p(h)\propto 1$。

While if another person also says "ignorant", but he uses $z=h^2$, $p(z)\propto 1$, then $p(h)\propto 2h$.

→ Bayesian 不能簡單說「不知道參數 ⇒ uniform prior」，要注意你說的是 uniform with respect to which parameterisation。

Jeffreys prior:

$$
\pi(\theta)\propto\sqrt{I(\theta)},\qquad I(\theta)=-E\left[\frac{\partial^2\log f(x\mid\theta)}{\partial\theta^2}\right]
$$

Invariant under reparameterisation，所以不管用哪種參數化，結果都一致。

---

## Part 2 Inversion sampling with MC method

If $X$ has cdf $F$, then $U=F(X)\sim U(0,1)$.

$$
\Pr(U\le u)=\Pr(X\le F^{-1}(u))=F(F^{-1}(u))=u
$$

所以反過來：draw $u\sim U(0,1)$，set $x=F^{-1}(u)$，則 $x\sim F$。

- 優點：independent samples
- 缺點：需要 $F^{-1}$ 算得出來；基本上只適用 univariate

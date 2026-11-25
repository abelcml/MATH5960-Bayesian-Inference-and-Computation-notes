## Part 1 MCMC with mixture models (Eyes data analysis)

### 1. Mixture density

每筆 $x_i$ 來自 $M$ 個 components 之一：

$$
f(x_i\mid\boldsymbol\theta,\boldsymbol\lambda)
=\sum_{m=1}^M\lambda_m f_m(x_i\mid\theta_m),
\qquad
\lambda_m\ge0,\quad\sum_{m=1}^M\lambda_m=1
$$

$f_m$ 是第 $m$ 個 component 的 density，$\theta_m$ 是它的 parameter vector。Components 常選同一個 distribution family，但 parameters 不同。$\lambda_m$ 是一筆 observation 來自 component $m$ 的機率。

### 2. Latent allocation

定義 $z_{im}=1$ 表示 $x_i$ 來自第 $m$ 個 component，否則為 0。每筆 observation 只有一個 label，所以 $\sum_{m=1}^M z_{im}=1$。

$$
\Pr(z_{im}=1\mid\boldsymbol\lambda)=\lambda_m,
\qquad
p(\boldsymbol z_i\mid\boldsymbol\lambda)
=\prod_{m=1}^M\lambda_m^{z_{im}}
$$

給定 weights，若各筆 allocation independent，則

$$
p(\boldsymbol z_1,\ldots,\boldsymbol z_n\mid\boldsymbol\lambda)
=\prod_{i=1}^n\prod_{m=1}^M\lambda_m^{z_{im}}
$$

---

## Part 2 Mixture model variant: Change-point models

---

## Part 3 Example: Reaction times and schizophrenia

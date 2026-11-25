## Part 1 Markov chains / MCMC

### 1. Definition

Markov property：給定 present state $X_t$，下一步的分布只依賴 $X_t$：

$$
\Pr(X_{t+1}\in A\mid X_0,\ldots,X_t)
=\Pr(X_{t+1}\in A\mid X_t)
$$

以下先用 discrete state space $E$ 和 time-homogeneous chain：

- $P(x,y)=\Pr(X_{t+1}=y\mid X_t=x)$：one-step transition probability。
- $P^n(x,y)=\Pr(X_n=y\mid X_0=x)$：$n$-step transition probability。
- $\pi$ 表示 stationary distribution，滿足 $\pi(y)=\sum_{x\in E}\pi(x)P(x,y)$。一般的 $X_t$ 分布會隨 $t$ 改變。

$n$ 步的 transition probability 要把所有 intermediate paths 加起來：

$$
P^n(x,y)=\sum_{x_1\in E}\cdots\sum_{x_{n-1}\in E}
P(x,x_1)P(x_1,x_2)\cdots P(x_{n-1},y)
$$

### 2. Example: Weather model

$E=\{R,S\}$（Rain, Sun），transition matrix 的 row / column 順序都是 $(R,S)$：

$$
P=\begin{pmatrix}0.4&0.6\\0.2&0.8\end{pmatrix}
$$

兩步只經過一個 intermediate state $x_1$：

$$
\begin{aligned}
P^2(R,S)
&=\sum_{x_1\in\{R,S\}}P(R,x_1)P(x_1,S)\\
&=\underbrace{P(R,R)P(R,S)}_{x_1=R}
+\underbrace{P(R,S)P(S,S)}_{x_1=S}\\
&=0.4(0.6)+0.6(0.8)=0.72
\end{aligned}
$$

三步則有兩個 intermediate states：

$$
P^3(R,S)=\sum_{x_1\in\{R,S\}}\sum_{x_2\in\{R,S\}}
P(R,x_1)P(x_1,x_2)P(x_2,S)
$$

一共有 $2^2=4$ 條 paths。

### 3. Irreducibility and aperiodicity

**Irreducible**：在 state space 內，任意 $x,y$ 都存在有限的 $n\ge0$，使 $P^n(x,y)>0$。Chain 可以走遍 state space，避免困在其中一個封閉 subset。

**Aperiodic**：定義 state $x$ 的 period：

$$
d_x=\gcd\{n\ge1:P^n(x,x)>0\}
$$

括號內是可能返回 $x$ 的步數。$d_x=1$ 時稱為 aperiodic。若可以在 2 或 3 步返回，也可以接起來走 $2+2=4$、$2+3=5$、$3+3=6$ 步，返回時間就包含 $\{2,3,4,5,\ldots\}$，gcd 是 1。

$P(x,x)>0$ 是 aperiodicity 的充分條件。即使沒有 one-step self-transition，也可能有 $d_x=1$。Irreducible chain 的所有 states 有相同 period。

### 4. Limit theorems

對 countable state space，irreducible、positive recurrent chain 有唯一 stationary distribution $\pi$。再加 aperiodicity，則 $P^n(x,\cdot)$ 收斂到 $\pi$。Finite irreducible chain 自動 positive recurrent。

Ergodic theorem 給 sample average 的 consistency。若 $E_\pi[|h(X)|]<\infty$，irreducible positive recurrent chain 滿足

$$
\frac1N\sum_{t=1}^N h(X_t)\xrightarrow{\mathrm{a.s.}}E_\pi[h(X)]
$$

這個 time-average 結果本身不需要 aperiodicity。Continuous state space 要用相應的 Harris recurrence 等條件，不能直接用「到某個單點的機率大於 0」定義 irreducibility。

### 5. Page 17: Why MCMC

設計一條以 posterior 為 stationary distribution、且符合收斂條件的 Markov chain，就能用長期 samples 估 posterior quantities。Samples 通常互相依賴。

Gibbs 和 MH 用來設計 transition rule；Monte Carlo 則用取得的 samples 算 mean、probability、credible intervals。

---

## Part 2 Gibbs sampling

### 1. Update full conditionals

Goal：從 joint posterior $\pi(\theta_1,\theta_2\mid x)$ 抽樣。雖然

$$
\pi(\theta_1,\theta_2\mid x)
=\pi(\theta_1\mid\theta_2,x)\pi(\theta_2\mid x)
$$

但 marginal $\pi(\theta_2\mid x)$ 可能不好抽。若兩個 full conditionals 容易抽，就可以交替更新。

選 initial state $\theta^{(0)}=(\theta_1^{(0)},\theta_2^{(0)})$，每輪 $t=1,2,\ldots$：

$$
\theta_1^{(t)}\sim\pi(\theta_1\mid\theta_2^{(t-1)},x),
\qquad
\theta_2^{(t)}\sim\pi(\theta_2\mid\theta_1^{(t)},x)
$$

更新 $\theta_2$ 時用剛抽到的 $\theta_1^{(t)}$。每一整輪留下 pair $(\theta_1^{(t)},\theta_2^{(t)})$。

Generalise 到 $p$ 個 parameters：每輪按 $i=1,\ldots,p$ 更新，

$$
\theta_i^{(t)}\sim\pi\left(
\theta_i\mid\theta_1^{(t)},\ldots,\theta_{i-1}^{(t)},
\theta_{i+1}^{(t-1)},\ldots,\theta_p^{(t-1)},x
\right)
$$

已更新的用本輪值，還沒更新的用上一輪值。符合收斂條件後，用這些 joint samples 做 posterior inference。

### 2. Example P25: Normal model

Goal：sample from $\pi(\mu,\sigma^2\mid x)$。假設 observations conditionally i.i.d.，且以下兩個 priors independent：

$$
X_i\mid\mu,\sigma^2\sim N(\mu,\sigma^2),
\qquad
\mu\sim N(\xi,\kappa^{-1}),
\qquad
\sigma^{-2}\sim\operatorname{Gamma}(\alpha,\beta)
$$

$\sigma^{-2}=1/\sigma^2$ 是 precision，Gamma 使用 shape–rate，$\kappa,\alpha,\beta>0$。

First，固定目前 precision，抽 $\mu$：

$$
\mu\mid\sigma^{-2},x\sim N\left(
\frac{\sigma^{-2}\sum_{i=1}^n x_i+\kappa\xi}{n\sigma^{-2}+\kappa},
\frac1{n\sigma^{-2}+\kappa}
\right)
$$

Second，使用剛抽到的 $\mu$，更新 precision：

$$
\sigma^{-2}\mid\mu,x\sim\operatorname{Gamma}\left(
\alpha+\frac n2,\;
\beta+\frac12\sum_{i=1}^n(x_i-\mu)^2
\right)
$$

### 3. P27–31: Failure of Gibbs

若 irreducibility 不成立，chain 可能困在某個 subset，從該起點無法探索完整 posterior。

### 4. P32: Slow convergence

![](data:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb%2813%2C%2013%2C%2013%29%22%20viewBox%3D%220%200%20340%20190%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20fill%3D%22none%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22M83%2090%20L154%2044%22%20stroke%3D%22%230D9488%22%2F%3E%3Cpath%20d%3D%22M83%2098%20L154%20150%22%20stroke%3D%22%2364748b%22%2F%3E%3Cpath%20d%3D%22M195%2044%20L268%2026%22%20stroke%3D%22%230D9488%22%2F%3E%3Cpath%20d%3D%22M195%2044%20L268%2074%22%20stroke%3D%22%2364748b%22%2F%3E%3Cpath%20d%3D%22M195%20150%20L268%20129%22%20stroke%3D%22%230D9488%22%2F%3E%3Cpath%20d%3D%22M195%20150%20L268%20173%22%20stroke%3D%22%2364748b%22%2F%3E%3C%2Fg%3E%3Crect%20x%3D%2221%22%20y%3D%2280%22%20width%3D%2264%22%20height%3D%2228%22%20rx%3D%227%22%20fill%3D%22%23D5EEE7%22%2F%3E%3Ctext%20x%3D%2253%22%20y%3D%2299%22%20text-anchor%3D%22middle%22%20font-size%3D%2212%22%20fill%3D%22%23243746%22%3E%CE%B8%E2%82%82%20%3D%201%3C%2Ftext%3E%3Crect%20x%3D%22141%22%20y%3D%2230%22%20width%3D%2264%22%20height%3D%2228%22%20rx%3D%227%22%20fill%3D%22%23E8EBF0%22%2F%3E%3Ctext%20x%3D%22173%22%20y%3D%2249%22%20text-anchor%3D%22middle%22%20font-size%3D%2212%22%20fill%3D%22%23243746%22%3E%CE%B8%E2%82%81%20%3D%201%3C%2Ftext%3E%3Crect%20x%3D%22141%22%20y%3D%22136%22%20width%3D%2264%22%20height%3D%2228%22%20rx%3D%227%22%20fill%3D%22%23E8EBF0%22%2F%3E%3Ctext%20x%3D%22173%22%20y%3D%22155%22%20text-anchor%3D%22middle%22%20font-size%3D%2212%22%20fill%3D%22%23243746%22%3E%CE%B8%E2%82%81%20%3D%200%3C%2Ftext%3E%3Crect%20x%3D%22265%22%20y%3D%2212%22%20width%3D%2264%22%20height%3D%2228%22%20rx%3D%227%22%20fill%3D%22%23D5EEE7%22%2F%3E%3Ctext%20x%3D%22297%22%20y%3D%2231%22%20text-anchor%3D%22middle%22%20font-size%3D%2212%22%20fill%3D%22%23243746%22%3E%CE%B8%E2%82%82%20%3D%201%3C%2Ftext%3E%3Crect%20x%3D%22265%22%20y%3D%2260%22%20width%3D%2264%22%20height%3D%2228%22%20rx%3D%227%22%20fill%3D%22%23E8EBF0%22%2F%3E%3Ctext%20x%3D%22297%22%20y%3D%2279%22%20text-anchor%3D%22middle%22%20font-size%3D%2212%22%20fill%3D%22%23243746%22%3E%CE%B8%E2%82%82%20%3D%200%3C%2Ftext%3E%3Crect%20x%3D%22265%22%20y%3D%22115%22%20width%3D%2264%22%20height%3D%2228%22%20rx%3D%227%22%20fill%3D%22%23D5EEE7%22%2F%3E%3Ctext%20x%3D%22297%22%20y%3D%22134%22%20text-anchor%3D%22middle%22%20font-size%3D%2212%22%20fill%3D%22%23243746%22%3E%CE%B8%E2%82%82%20%3D%201%3C%2Ftext%3E%3Crect%20x%3D%22265%22%20y%3D%22159%22%20width%3D%2264%22%20height%3D%2228%22%20rx%3D%227%22%20fill%3D%22%23E8EBF0%22%2F%3E%3Ctext%20x%3D%22297%22%20y%3D%22178%22%20text-anchor%3D%22middle%22%20font-size%3D%2212%22%20fill%3D%22%23243746%22%3E%CE%B8%E2%82%82%20%3D%200%3C%2Ftext%3E%3Cg%20fill%3D%22currentColor%22%20font-size%3D%2212%22%20text-anchor%3D%22middle%22%3E%3Ctext%20x%3D%22115%22%20y%3D%2255%22%3Ep%3C%2Ftext%3E%3Ctext%20x%3D%22112%22%20y%3D%22138%22%3E1%E2%88%92p%3C%2Ftext%3E%3Ctext%20x%3D%22229%22%20y%3D%2224%22%3Ep%3C%2Ftext%3E%3Ctext%20x%3D%22232%22%20y%3D%2272%22%3E1%E2%88%92p%3C%2Ftext%3E%3Ctext%20x%3D%22230%22%20y%3D%22122%22%3E1%E2%88%92p%3C%2Ftext%3E%3Ctext%20x%3D%22230%22%20y%3D%22175%22%3Ep%3C%2Ftext%3E%3C%2Fg%3E%3C%2Fsvg%3E)

Parameters highly correlated 時，逐一更新可能每次只移動一小段，mixing 很慢。

---

## Part 3 Assessing convergence

### 1. Burn-in

丟掉早期受 initial state 影響較大的 samples。Burn-in 長度要配合 diagnostics；丟掉固定幾步本身不保證收斂。

### 2. Thinning

每隔 $k$ 步存一筆，主要用來減少 storage。留下的 samples 仍可能 autocorrelated；固定運算成本下，thinning 通常會丟掉可用資訊。

### 3. Multiple chains

從 dispersed starting points 跑多條 chains，用 $\hat R$ 比較 within-chain 和 between-chain variation。$\hat R$ 接近 1 是診斷訊號，也要一起看 trace plots、effective sample size 和 MCSE；單一 diagnostic 無法證明已收斂。

### 4. Blocking

Highly correlated parameters 可以放在同一個 block，從 joint conditional 一起更新，減少逐座標移動造成的停滯。

![[Pasted image 20261016174438.png|331]]

### 5. Reparameterisation

當 $\operatorname{Corr}(\theta_1,\theta_2)<0$ 且相關很強，Gibbs 的逐座標更新容易沿細長的 posterior region 走 zigzag。

![](data:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb%2813%2C%2013%2C%2013%29%22%20viewBox%3D%220%200%20330%20195%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20transform%3D%22translate%28165%2097%29%20rotate%28-44%29%22%3E%3Cellipse%20cx%3D%220%22%20cy%3D%220%22%20rx%3D%2238%22%20ry%3D%229.5%22%20fill%3D%22none%22%20stroke%3D%22%2387919E%22%20stroke-opacity%3D%22.55%22%20stroke-width%3D%221.2%22%2F%3E%3Cellipse%20cx%3D%220%22%20cy%3D%220%22%20rx%3D%2258%22%20ry%3D%2214.5%22%20fill%3D%22none%22%20stroke%3D%22%2387919E%22%20stroke-opacity%3D%22.55%22%20stroke-width%3D%221.2%22%2F%3E%3Cellipse%20cx%3D%220%22%20cy%3D%220%22%20rx%3D%2278%22%20ry%3D%2219.5%22%20fill%3D%22none%22%20stroke%3D%22%2387919E%22%20stroke-opacity%3D%22.55%22%20stroke-width%3D%221.2%22%2F%3E%3Cellipse%20cx%3D%220%22%20cy%3D%220%22%20rx%3D%2297%22%20ry%3D%2224.25%22%20fill%3D%22none%22%20stroke%3D%22%2387919E%22%20stroke-opacity%3D%22.55%22%20stroke-width%3D%221.2%22%2F%3E%3Cellipse%20cx%3D%220%22%20cy%3D%220%22%20rx%3D%22115%22%20ry%3D%2228.75%22%20fill%3D%22none%22%20stroke%3D%22%2387919E%22%20stroke-opacity%3D%22.55%22%20stroke-width%3D%221.2%22%2F%3E%3C%2Fg%3E%3Cpath%20d%3D%22M18%20176%20H312%20M18%20176%20V12%22%20stroke%3D%22currentColor%22%20stroke-width%3D%221.3%22%20fill%3D%22none%22%2F%3E%3Cpolyline%20points%3D%2285%2C36%20110%2C36%20110%2C62%20143%2C62%20143%2C95%20180%2C95%20180%2C126%20221%2C126%20221%2C157%22%20stroke%3D%22%23CC5268%22%20stroke-width%3D%222.5%22%20fill%3D%22none%22%2F%3E%3Ctext%20x%3D%22306%22%20y%3D%22189%22%20text-anchor%3D%22end%22%20font-size%3D%2212%22%20fill%3D%22currentColor%22%3E%CE%B8%E2%82%81%3C%2Ftext%3E%3Ctext%20x%3D%228%22%20y%3D%2215%22%20font-size%3D%2212%22%20fill%3D%22currentColor%22%3E%CE%B8%E2%82%82%3C%2Ftext%3E%3C%2Fsvg%3E)

換一組 parameterisation，降低 posterior dependence，可能改善 mixing；不一定能做到完全 independent。

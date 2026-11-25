## Part 1 Topic of priors

### 1.1 Conjugate prior

**一句話：prior 與 posterior 留在同一個 distribution family，更新方便。**

| Likelihood | Conjugate prior |
| --- | --- |
| Binomial 機率 | Beta |
| Poisson rate | Gamma |
| Normal mean（variance 已知） | Normal |
| Gamma likelihood 的 rate（shape 已知） | Gamma |

Normal mean 的直覺：**posterior precision = prior precision + data precision**。

若 likelihood 是 $f(x\mid\theta)=h(x)g(\theta)\exp[t(x)c(\theta)]$，可考慮同族先驗 $\pi(\theta)\propto g(\theta)^d e^{bc(\theta)}$（hyperparameters 要讓 prior proper）。

### 1.2 Improper priors

- **Proper**：prior 積分 = 1。
- **Improper**：prior 無法積分成 1；仍可用，但 **posterior 必須 proper**。
- 例：$\pi(\mu)\propto1$ 在 $\mu\in\mathbb R$ 上 improper；$U(0,1)$ proper。

### 1.3 Jeffreys priors

**問題：**「不知道」不代表可以隨便設 uniform；換參數後 uniform 不一定還是 uniform。

$$
\boxed{\pi_J(\theta)\propto\sqrt{I(\theta)}}
$$

$I(\theta)=-E[\partial^2\log L(x\mid\theta)/\partial\theta^2]$ 是 Fisher information（用一筆或整份資料只差不依賴 $\theta$ 的常數）。多維用 $\sqrt{\det I(\theta)}$。

**重點：** Jeffreys prior 對 smooth one-to-one reparameterisation 具 invariant 性質，仍可能 improper。

---

## Part 2 Inversion sampling with MC method

**目的：** 將 uniform random number 變成目標分布的 sample。

$$
\boxed{U\sim U(0,1),\quad X=F^{-1}(U)\ \Rightarrow\ X\sim F}
$$

$F$ 是 target CDF。若 $F$ 非嚴格遞增，用 generalized inverse：$F^{-1}(u)=\inf\{x:F(x)\ge u\}$。

- **好處**：uniforms 獨立 → samples 獨立。
- **困難**：要能算出 inverse CDF；實務上主要用於 univariate。

**考試要會**

分辨 conjugate / improper / Jeffreys，以及寫出 $U\to F^{-1}(U)$ 的 inversion algorithm。

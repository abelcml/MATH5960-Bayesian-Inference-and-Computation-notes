## 1. Bayesian hypothesis testing

兩個 hypotheses，資料比較支持哪一個？

![[Pasted image 20261115232714.png]]

---

## 2. Bayesian model comparison

---

## 3. Bayes factor

Want to compare models given $x$。令 $\pi(i)$ 為 model $M_i$ 的 prior probability，$m_i(x)=p(x\mid M_i)$ 為 marginal likelihood：

$$
m_i(x)=\int p(x\mid\theta_i,M_i)\pi(\theta_i\mid M_i)\,d\theta_i
$$

Bayes' theorem 給 $\pi(i\mid x)=m_i(x)\pi(i)/m(x)$，因此

$$
\frac{\pi(i\mid x)}{\pi(j\mid x)}
=\frac{m_i(x)\pi(i)/m(x)}{m_j(x)\pi(j)/m(x)}
=\frac{\pi(i)}{\pi(j)}B_{ij},
\qquad
B_{ij}=\frac{m_i(x)}{m_j(x)}
$$

Posterior odds = prior odds × Bayes factor。$B_{ij}>1$ 表示資料讓 odds 朝 $M_i$ 的方向增加；最後 posterior odds 還取決於 prior odds。

一般的 marginal likelihood 比較需要 proper parameter priors，否則 improper prior 的任意常數會使 Bayes factor 無法唯一決定。

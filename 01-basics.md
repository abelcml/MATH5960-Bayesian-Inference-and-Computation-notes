# 01 Basics

## 1. One word explain Bayesian

Parameters are represented by distributions, not point estimates.

Three components: prior, likelihood, posterior.

## 2. Unknown colored ball in bag

The prior problem is how much possible there are 3 black balls in bag.

## 3. Usual distributions form

| 看到這個形狀（對參數而言） | 就是 |
| --- | --- |
| $\exp\left(-(\theta-m)^2/2v\right)$ | Normal |
| $\theta^{a-1}e^{-b\theta}$ | Gamma |
| $(\sigma^2)^{-(a+1)}e^{-b/\sigma^2}$ | Inverse-Gamma |
| $\theta^{a-1}(1-\theta)^{b-1}$ | Beta |
| $(1+t^2/\nu)^{-(\nu+1)/2}$ | t |

Bayesian 這門課大部分推導都是「乘起來 → 丟掉常數 → 認核」，認得這張表，八成的推導就不會再卡住了。


## one word explain Bayesian
Parameters are represented by distributions, not point estimates
three components: prior , likelihood , posterior

## unknown colored ball in bag
the prior problem is how much possible there're are 3 black balls in bag


## usual distributions form
| 看到这个形状（对参数而言）                        | 就是            |
| ------------------------------------ | ------------- |
| $\exp(-(\theta-m)^2/2v)$             | Normal        |
| $\theta^{a-1}e^{-b\theta}$           | Gamma         |
| $(\sigma^2)^{-(a+1)}e^{-b/\sigma^2}$ | Inverse-Gamma |
| $\theta^{a-1}(1-\theta)^{b-1}$       | Beta          |
| $(1+t^2/\nu)^{-(\nu+1)/2}$           | t             |

Bayesian 这门课大部分推导都是「乘起来 → 丢掉常数 → 认核」，认得这张表，八成的推导就不会再卡住了。
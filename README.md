Data Generator Engine
=====================

Data Generator Engine (DGE) generates new data that is based on existing data. Therefore, DGE models the distribution
of the input data. The underlying distribution is used to sample new data based on the given and existing data.

The current implementation of DGE is written as Python library, which allows the integration of DGE in Python projects.
DGE uses NumPy and SciPy to do the required calculations and scikit-learn to implement the machine learning.

## Univariate distributions
Univariate distributions are defined for one-dimensional random variables. The random variables are continuous, which
means that the values are not discrete. The values are typically contained in the set of real values.
Example of sampling random variates from an univariate probability distribution:

```python
import dge
ud = UnivariateData()

import numpy as np
import matplotlib.pyplot as plt
d = np.random.normal(20.0, 1.0, 1000)
ud.fit(d)
dhat = ud.sample(1000)
print("Mean = {}, STD = {}".format(dhat.mean(),dhat.std()))

d = np.random.beta(2,5,1000)
ud.fit(d)
dhat = ud.sample(1000)
plt.hist(d) and plt.hist(dhat) and plt.show()
```

DGE models the univariate distributions by estimating the cumulative distribution function (CDF) of the data using
regression methods. The random variates are sampled using an uniform distribution and the estimated CDF. See the paper
[Neural Networks for Density Estimation](http://papers.nips.cc/paper/1624-neural-networks-for-density-estimation.pdf) for further information.

### Read 10 random variates from a stream
```python
for random_variate in ud.stream(10):
    print(random_variate)
```

## Multivariate distributions
### Multivariate Correlated Data (MCD)
MCD models the correlation structure of the observed data. MCD uses Cholesky decomposition of the correlation
matrix to generate random variates that have the observed correlation structure. The following example samples
random variates with independent variables from Gaussian distribution. MCD models the hypothetical correlation
structure relatively well.

```python
import dge
import numpy as np
mcd = dge.MultivariateCorrelatedData()
d = np.random.normal(0, 4, (100, 3))
print(np.corrcoef(d, rowvar=False))

    array([[ 1.        , -0.05467565,  0.05282523],
           [-0.05467565,  1.        ,  0.02704508],
           [ 0.05282523,  0.02704508,  1.        ]])

mcd.fit(d)
print(np.corrcoef(np.array(list(mcd.stream(100))), rowvar=False))

    array([[ 1.        , -0.09294713,  0.01578568],
       [-0.09294713,  1.        ,  0.10180806],
       [ 0.01578568,  0.10180806,  1.        ]])

```

## Discrete distributions
The current implementation is a simple frequentist approach where the probability of an observed event is:
* Independent of the previous observations
* Measured as the fraction of the number of observations and the number of all observations
```python
import dge
dd = dge.DiscreteData()
dd.fit([0,0,1])
dd.sample()
```

Consider a situation where you observe a series of two events: 0, 0 and 1. Therefore, from the frequentist point of
view, there is 66% probability of observing the event 0 and 33% probability of observing the event 1. Now we will
sample random variates as series of length from 2 to 300:
```python
for i in range(2,301):
    dd.stream(i)
```

We will calculate the probabilities of the observation 0 for every series and the mean of the observed probabilities.
The mean is called the mean of the sampling distribution of the mean. Central Limit Theorem states that as the number
of series approaches infinity, the mean of the sampling distribution of the mean approaches the population mean. The
following figure illustrates the results:
![alt](https://github.com/nikoreun/data-generator-engine/raw/master/markdown/dge_discrete_example.png)
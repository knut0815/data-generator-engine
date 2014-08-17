data-generator-engine
=====================

Data Generator Engine (DGE) generates new data that is based on existing data. Therefore, DGE models the distribution
of the input data. The underlying distribution is used to sample new data based on the given and existing data.

The current implementation of DGE is written as Python library, which allows the integration of DGE in Python projects.
DGE uses NumPy and SciPy to do the required calculations and scikit-learn to implement the machine learning.

Example of sampling random variates from an univariate probability distribution:

```python
import dge
import numpy as np
import matplotlib.pyplot as plt
d = np.random.normal(20.0, 1.0, 1000)
ud = UnivariateData()
ud.fit(d)
dhat = ud.sample(1000)
print("Mean = {}, STD = {}".format(dhat.mean(),dhat.std()))

d = np.random.beta(2,5,1000)
ud.fit(d)
dhat = ud.sample(1000)
plt.hist(d) and plt.hist(dhat) and plt.show()

# Read 10 random variates from a stream
for random_variate in ud.stream(10):
    print(random_variate)
```
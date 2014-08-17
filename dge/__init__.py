__author__ = 'Niko Reunanen'
from sklearn.ensemble import RandomForestRegressor as RFR
import numpy as np


class DataGenerator(object):
    def __init__(self):
        self.ready = False

    def fit(self, x):
        pass

    def sample(self):
        return 0.0

    def stream(self, max_=None):
        """
        Stream the random variates as Python generator.
        :param max_: Maximum number of stream elements (default None is infinite)
        :return: A generator of random variates
        """
        while max_ is None or max_ > 0:
            if max_ is not None:
                max_ -= 1
            yield(self.sample())


class DiscreteData(DataGenerator):
    def __init__(self):
        super(DataGenerator, self).__init__()
        self.events = None
        self.p = None
        self.p_cumsum = None

    def fit(self, x):
        """
        Fit the data generator to produce random observations that follow the distribution
        of the input data 'x'. The modeled distribution is based on the given observations.
        :param x: NumPy array of list of observed events
        :return: None
        """
        x = np.array(x)
        self.events = np.unique(x)
        self.p = np.bincount(x) / float(x.size)
        self.p_cumsum = np.cumsum(self.p)

    def sample(self):
        """
        The method returns a random variate that follows the fitted distribution.
        :return:An observed event
        :rtype:int
        """
        t_ = np.random.uniform(0, 1)
        for e, p in zip(self.events, self.p_cumsum):
            if t_ <= p:
                return e
        return None


class MultivariateCorrelatedData(DataGenerator):
    def __init__(self):
        """
        MultivariateCorrelatedData models the correlations between the random variables of a given data set.
        :return: None
        """
        super(DataGenerator, self).__init__()
        self.L = None
        self.correlation_matrix = None
        self.max_ = None
        self.min_ = None

    def fit(self, x):
        """
        Fit the data generator to produce random variates that follow the distribution of the input data 'x'.
        :param x: Data set as a n*d matrix (n observations, d dimensions)
        :return: None
        """
        x = np.array(x)
        self.correlation_matrix = np.corrcoef(x, rowvar=False)
        self.L = np.linalg.cholesky(self.correlation_matrix)
        self.max_ = np.max(x, axis=0)
        self.min_ = np.min(x, axis=0)

    def sample(self):
        """
        Sample a random variate that follows the correlation structure of the observed data.
        :return: A random variate as a NumPy array
        """
        return np.dot(np.random.uniform(self.min_, self.max_), self.L)


class UnivariateData(DataGenerator):
    def __init__(self, regression=RFR):
        """
        UnivariateData models the cumulative probability distribution (CDF) of an arbitrary univariate distribution. The
        modeled distribution is used to sample random variates. CDF is modeled using regression.
        :param regression: scikit-learn regressor that implements methods "fit" and "predict"
        :return: None
        """
        super(DataGenerator, self).__init__()
        self.regression = regression()

    def fit(self, x):
        """
        Fit the data generator to produce random variates that follow the distribution
        of the input data 'x'. The modeled distribution is an one-dimensional probability
         distribution.
        :param x: NumPy array or list
        :return: None
        """
        x = np.array(x).reshape((x.size, ))
        x.sort()
        y = np.random.uniform(0, 1, (x.size, 1))
        y.sort()
        self.regression.fit(y, x)

    def sample(self, n=1):
        """
        The method returns a random variate that follows the fitted distribution.
        :return:Numpy array of random variates
        :rtype:numpy.ndarray
        """
        return self.regression.predict(np.random.uniform(0, 1, (n, 1)))
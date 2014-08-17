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


class UniformData(DataGenerator):
    def __init__(self, regression=RFR):
        """
        UniformData models the cumulative probability distribution (CDF) of an arbitrary uniform distribution. The
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
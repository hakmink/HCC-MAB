import abc
import numpy as np


class Arm(object):
    """
    An interface of arms (of the bandits in the multi-armed bandit)
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def draw(self):
        pass

    @abc.abstractmethod
    def get_expected_value(self):
        pass


class BinomialArm(Arm):
    """
    Arm that is from the binomial distribution with parameters n and p, is the discrete
    probability distribution of the number of successes in a sequence of n independent experiments.
    """
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def draw(self):
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self):
        return self.n * self.p

    def __str__(self):
        return 'Bin(n={},p={})'.format(self.n, self.p)


class BernoulliArm(BinomialArm):
    """
    Arm that has the probability of getting a reward of 1 from that arm.
    The Bernoulli distribution is the special case of the binomial distribution when n=1.
    """
    def __init__(self, p):
        super(BernoulliArm, self).__init__(1, p)

    def get_expected_value(self):
        return self.p

    def __str__(self):
        return 'Bern(p={})'.format(self.p)


class GaussianArm(Arm):
    """
    Arm that is from the normal(aka Gaussian) distribution with parameters mu and sigma,
    are often used in the natural and social sciences to represents real-valued random variables
    whose distributions are not known.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def draw(self):
        return np.random.normal(self.mu, self.sigma)

    def get_expected_value(self):
        return self.mu

    def __str__(self):
        return 'Norm(mu={},sigma={})'.format(self.mu, self.sigma)

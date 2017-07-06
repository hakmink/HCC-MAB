import abc
import numpy as np
from scipy.special import expit
from scipy.optimize import fmin_bfgs
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class ContextualBanditAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_arm(self, context):
        pass

    @abc.abstractmethod
    def update(self, arm, reward):
        pass

    @abc.abstractmethod
    def add_arm(self):
        pass

    @abc.abstractmethod
    def remove_arm(self, arm):
        pass


# Lihong Li et al. (2010)
# dim: number of article features, alpha: coefficient of confidence bound
class LinUCBDisjointAlgorithm(ContextualBanditAlgorithm):
    def __init__(self, num_arms, num_dimensions, alpha):
        self.num_arms = num_arms
        self.num_dimensions = num_dimensions
        self.alpha = alpha
        self.p = np.ones(num_arms) / num_arms
        self.context = None

        self.A = np.tile(np.identity(num_dimensions), (num_arms, 1)).reshape(num_arms, num_dimensions, num_dimensions)
        self.A_inv = self.A.copy()
        self.b = np.zeros(shape=(num_arms, num_dimensions))
        """
        # same to above (but much more comprehensive)
        self.A = np.array([np.identity(num_dimensions) for i in range(num_arms)])  # multiple A's
        self.A_inv = np.array([np.identity(num_dimensions) for i in range(num_arms)])  # A^-1 cache
        self.b = np.array([np.zeros(num_dimensions) for i in range(num_arms)])  # multiple b's
        """

    def select_arm(self, context):
        # context: num_arms * num_dimensions numpy ndarray
        self.context = context
        for arm in range(self.num_arms):
            theta = self.A_inv[arm].dot(self.b[arm])  # theta = A^-1 b
            # p_a = t(theta_a) x_a + alpha sqrt {t(x_a) A^-1 x_a}, where t(): transpose
            self.p[arm] = theta.dot(context[arm]) +\
                self.alpha * np.sqrt(context[arm].dot(self.A_inv[arm]).dot(context[arm]))
        return np.random.choice(np.flatnonzero(self.p == self.p.max()))  # tie-breaking

    def update(self, arm, reward):
        self.A[arm] += np.outer(self.context[arm], self.context[arm])
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.b[arm] += reward * self.context[arm]

    def add_arm(self):
        self.num_arms += 1
        self.A = np.append(self.A, np.identity(self.num_dimensions)).reshape(
            (self.num_arms, self.num_dimensions, self.num_dimensions))
        self.A_inv = np.append(self.A_inv, np.identity(self.num_dimensions)).reshape(
            (self.num_arms, self.num_dimensions, self.num_dimensions))
        self.b = np.vstack((self.b, np.zeros(self.num_dimensions)))
        self.p = np.append(self.p, np.zeros(1))

    def remove_arm(self, arm):
        self.num_arms -= 1
        d = self.num_dimensions
        d2 = self.num_dimensions * self.num_dimensions
        # removing elements of A[arm*d^2 ~ (arm+1)*d^2-1] occurs array flattening -> reshaping required
        self.A = np.delete(self.A, np.arange(arm * d2, (arm + 1) * d2)).reshape((self.num_arms, d, d))
        self.A_inv = np.delete(self.A_inv, np.arange(arm * d2, (arm + 1) * d2)).reshape((self.num_arms, d, d))
        self.b = np.delete(self.b, np.arange(arm * d, (arm + 1) * d)).reshape(self.num_arms, d)
        self.p = np.delete(self.p, arm)

    def __str__(self):
        return 'LinUCBDisjoint(alpha={})'.format(self.alpha)


# BHOH Original
# IT WORKS VERY WELL WHY? :D
class FastLinUCBDisjointAlgorithm(LinUCBDisjointAlgorithm):
    def update(self, arm, reward):
        inc_a = np.outer(self.context[arm], self.context[arm])
        self.A[arm] += inc_a
        self.A_inv[arm] = self.A_inv[arm] - self.A_inv[arm] * inc_a * self.A_inv[arm]
        self.b[arm] += reward * self.context[arm]

    def __str__(self):
        return 'FastLinUCBDisjoint(alpha={})'.format(self.alpha)


# BHOH Original 2
# very disappointing...
class FastLinUCBDisjointAlgorithm2(LinUCBDisjointAlgorithm):
    def select_arm(self, context):
        self.context = context
        for arm in range(self.num_arms):
            theta = self.A_inv[arm].dot(self.b[arm])
            self.p[arm] = theta.dot(context[arm]) +\
                self.alpha * np.sqrt(context[arm].dot(self.A_inv[arm]).dot(context[arm]).clip(min=0))
        return np.random.choice(np.flatnonzero(self.p == self.p.max()))  # tie-breaking

    def update(self, arm, reward):
        inc_a = np.outer(self.context[arm], self.context[arm])
        self.A[arm] += 0.01 * inc_a
        self.A_inv[arm] -= 0.01 * inc_a
        self.b[arm] += reward * self.context[arm]

    def __str__(self):
        return 'FastLinUCBDisjoint(alpha={})'.format(self.alpha)


# Lihong Li et al. (2010)
# k: number of hybrid features, d: number of article features, alpha: coefficient of confidence bound
class LinUCBHybridAlgorithm(ContextualBanditAlgorithm):
    def __init__(self, num_arms, num_dimensions_k, num_dimensions_d, alpha):
        self.num_arms = num_arms
        self.num_dim_k = num_dimensions_k
        self.num_dim_d = num_dimensions_d
        self.alpha = alpha
        self.p = np.ones(num_arms) / num_arms
        self.z = None  # context z: hybrid features
        self.x = None  # context x: article features
        self.A0 = np.identity(num_dimensions_k)
        self.b0 = np.zeros(num_dimensions_k)

        self.A = np.tile(np.identity(num_dimensions_d), (num_arms, 1)).reshape(
            num_arms, num_dimensions_d, num_dimensions_d)
        self.A_inv = self.A.copy()
        self.B = np.tile(np.zeros(shape=(num_dimensions_d, num_dimensions_k)), (num_arms, 1)).reshape(
            num_arms, num_dimensions_d, num_dimensions_k)
        self.b = np.zeros(shape=(num_arms, num_dimensions_d))
        """
        # same to above (but much more comprehensive)
        self.A = np.array([np.identity(num_dimensions_d) for i in range(num_arms)])  # multiple A's
        self.A_inv = np.array([np.identity(num_dimensions_d) for i in range(num_arms)])  # A^-1 cache
        self.B = np.array([np.zeros(shape=(num_dimensions_d, num_dimensions_k)) for i in range(num_arms)])  # B's
        self.b = np.array([np.zeros(num_dimensions_d) for i in range(num_arms)])  # b's
        """

    def select_arm(self, context):
        # context: num_arms * (num_dimensions_k + num_dimensions_d) numpy ndarray
        self.z = context[:, :self.num_dim_k]
        self.x = context[:, self.num_dim_k:]
        a0_inv = np.linalg.inv(self.A0)
        beta = a0_inv.dot(self.b0)
        for arm in range(self.num_arms):
            theta = self.A_inv[arm].dot(self.b[arm] - self.B[arm].dot(beta))
            z_t_a0_inv = self.z[arm].dot(a0_inv)
            a_inv_x = self.A_inv[arm].dot(self.x[arm])
            s = z_t_a0_inv.dot(self.z[arm]) - 2 * z_t_a0_inv.dot(self.B[arm].T).dot(a_inv_x) +\
                self.x[arm].dot(a_inv_x) +\
                self.x[arm].dot(self.A_inv[arm]).dot(self.B[arm]).dot(a0_inv).dot(self.B[arm].T).dot(a_inv_x)
            self.p[arm] = self.z[arm].dot(beta) + self.x[arm].dot(theta) + self.alpha * np.sqrt(s)
        return np.random.choice(np.flatnonzero(self.p == self.p.max()))  # tie-breaking

    def update(self, arm, reward):
        b_t_a_inv = self.B[arm].T.dot(self.A_inv[arm])
        self.A0 += b_t_a_inv.dot(self.B[arm])
        self.b0 += b_t_a_inv.dot(self.b[arm])
        self.A[arm] += np.outer(self.x[arm], self.x[arm])
        self.B[arm] += np.outer(self.x[arm], self.z[arm])
        self.b[arm] += reward * self.x[arm]
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        b_t_a_inv = self.B[arm].T.dot(self.A_inv[arm])
        self.A0 += np.outer(self.z[arm], self.z[arm]) - b_t_a_inv.dot(self.B[arm])
        self.b0 += reward * self.z[arm] - b_t_a_inv.dot(self.b[arm])

    def add_arm(self):
        self.num_arms += 1
        self.A = np.append(self.A, np.identity(self.num_dim_d)).reshape(
            (self.num_arms, self.num_dim_d, self.num_dim_d))
        self.A_inv = np.append(self.A_inv, np.identity(self.num_dim_d)).reshape(
            (self.num_arms, self.num_dim_d, self.num_dim_d))
        self.B = np.append(self.B, np.zeros(shape=(self.num_dim_d, self.num_dim_k))).reshape(
            (self.num_arms, self.num_dim_d, self.num_dim_k))
        self.b = np.vstack((self.b, np.zeros(self.num_dim_d)))
        self.p = np.append(self.p, np.zeros(1))

    def remove_arm(self, arm):
        self.num_arms -= 1
        d = self.num_dim_d
        d2 = self.num_dim_d * self.num_dim_d
        dk = self.num_dim_d * self.num_dim_k
        # removing elements of A[arm*d^2 ~ (arm+1)*d^2-1] occurs array flattening -> reshaping required
        self.A = np.delete(self.A, np.arange(arm * d2, (arm + 1) * d2)).reshape((self.num_arms, d, d))
        self.A_inv = np.delete(self.A_inv, np.arange(arm * d2, (arm + 1) * d2)).reshape((self.num_arms, d, d))
        self.B = np.delete(self.B, np.arange(arm * dk, (arm + 1) * dk)).reshape((self.num_arms, d, self.num_dim_k))
        self.b = np.delete(self.b, np.arange(arm * d, (arm + 1) * d)).reshape(self.num_arms, d)
        self.p = np.delete(self.p, arm)

    def __str__(self):
        return 'LinUCBHybrid(alpha={})'.format(self.alpha)


# Agrawal and Goyal (2013 ICML, 2014 arXiv)
# R and delta determines v, which is a magnitude of covariance B_inv
class LinThompsonAlgorithm(ContextualBanditAlgorithm):
    """
    warning: not working well on yahoo! news
    cause: self.f is not updated when reward=0 (with p(reward=1)~=1/50)
    """
    def __init__(self, num_arms, num_dimensions, r=0.5, delta=0.05):
        self.num_arms = num_arms
        self.num_dimensions = num_dimensions
        self.r = r
        self.delta = delta

        # self.v2 = R * R * (24.0 / epsilon * num_dimensions * np.log(1.0 / delta))  # ICML 2013, fixed parameter
        self.vt2_multiplier_of_lnt = 9 * num_dimensions * r * r
        self.vt2_subtraction = self.vt2_multiplier_of_lnt * np.log(delta)
        self.t = 1  # current time

        self.B = np.identity(num_dimensions)
        self.B_inv = np.identity(num_dimensions)
        self.mu = np.zeros(num_dimensions)
        self.f = np.zeros(num_dimensions)
        self.context = None

    def select_arm(self, context):
        self.context = context
        vt2 = self.vt2_multiplier_of_lnt * np.log(self.t) - self.vt2_subtraction  # arXiv 2014
        sample_mu = np.random.multivariate_normal(self.mu, vt2 * self.B_inv)
        p = context.dot(sample_mu)
        is_max = np.array(p == p.max())
        return np.random.choice(np.flatnonzero(is_max))  # tie-breaking

    def update(self, arm, reward):
        self.t += 1
        self.B += np.outer(self.context[arm], self.context[arm])
        self.f += reward * self.context[arm]
        self.B_inv = np.linalg.inv(self.B)
        self.mu = self.B_inv.dot(self.f)

    def add_arm(self):
        self.num_arms += 1

    def remove_arm(self, arm):
        self.num_arms -= 1

    def __str__(self):
        return 'LinThompson(R={},delta={})'.format(self.r, self.delta)


# Chapelle and Li (2011)
# reg_lambda controls the balance between regression accuracy and parameter consistency
# size_batch is the number of samples in mini-batches
class LogRegThompsonAlgorithm(ContextualBanditAlgorithm):
    """
    warning: not working well on yahoo! news
    cause: extremely unbalanced data with the proportion of reward, reward 0: reward 1 = 23:1
    """
    def __init__(self, num_arms, num_dimensions, reg_lambda=0.5, size_batch=10):
        self.num_arms = num_arms
        self.num_dimensions = num_dimensions
        self.reg_lambda = reg_lambda
        self.size_batch = size_batch
        self.m = np.zeros(num_dimensions)
        self.q = reg_lambda * np.ones(num_dimensions)
        self.context = None
        self.num_collected_batch = 0
        self.context_batch = np.zeros(shape=(num_dimensions, size_batch))
        self.reward_batch = np.zeros(size_batch)

    def select_arm(self, context):
        self.context = context
        w = np.random.multivariate_normal(self.m, np.diag(1.0 / self.q))
        p = expit(w.dot(context.T))  # sigmoid of wx
        return np.random.choice(np.flatnonzero(p == p.max()))  # tie-breaking

    def update(self, arm, reward):
        self.context_batch[:, self.num_collected_batch] = self.context[arm]
        self.reward_batch[self.num_collected_batch] = reward
        self.num_collected_batch += 1
        if self.num_collected_batch == self.size_batch:
            # minimization w/ BFGS (commonly used for fitting logistic models)
            self.m = fmin_bfgs(self.__func_to_minimize, np.zeros(self.num_dimensions), disp=False)
            p = expit(np.dot(self.m, self.context_batch))
            self.q += np.sum(np.power(self.context_batch, 2).dot(p * (1 - p)))
            self.num_collected_batch = 0

    def add_arm(self):
        self.num_arms += 1

    def remove_arm(self, arm):
        self.num_arms -= 1

    def __str__(self):
        return 'LogRegThompson(lambda={},batch={})'.format(self.reg_lambda, self.size_batch)

    def __func_to_minimize(self, w):
        reg = 0.5 * np.sum(self.q * np.power(w - self.m, 2))
        return reg + np.sum(np.log(1.0 + np.exp(-self.reward_batch * w.dot(self.context_batch))))


# Srinivas et al. (2010)
# GP-UCB corresponds to the disjoint version of LinUCB (considers only one feature space only)
# warning: the input space D={x1, x2, ...} gets bigger and bigger... (and the algorithm gets slower and slower...)
class GPUCBAlgorithm(ContextualBanditAlgorithm):
    """
    warning: not working well on yahoo! news
    cause: extremely unbalanced data with the proportion of reward, reward 0: reward 1 = 23:1
    """
    def __init__(self, num_arms, num_dimensions, kernel=None, beta=0.1):
        self.num_arms = num_arms
        self.num_dimensions = num_dimensions
        self.kernel = kernel
        if kernel is None:
            self.kernel = RBF() + WhiteKernel()
        self.beta_fixed = False
        self.beta = None
        if beta is not None:
            self.beta = beta
            self.beta_fixed = True
        self.K_inv = None
        self.init = True
        self.context = None
        self.context_collection = None
        self.rewards = np.array([])
        self.t = 1
        self.hyper_const = 1  # in theorem 2 of Srinivas et al., abstraction of a, b, delta, r

    def select_arm(self, context):
        self.context = context
        if self.init:
            return np.random.randint(self.num_arms)

        # from "Gaussian Process for Regression: A Quick Introduction" by M. Ebden
        gp_k_star = self.kernel(self.context, self.context_collection)
        gp_k_star2 = self.kernel(self.context)
        gp_k_star_mul_k_inv = gp_k_star.dot(self.K_inv)
        mu = gp_k_star_mul_k_inv.dot(self.rewards)
        variance = gp_k_star2 - gp_k_star_mul_k_inv.dot(gp_k_star.T)

        self.__update_beta()
        p = mu + np.sqrt(self.beta * np.diag(variance))  # original strategy of choosing arm
        # p = np.random.multivariate_normal(p, variance)  # sampling of GPR
        is_max = np.array(p == p.max())
        return np.random.choice(np.flatnonzero(is_max))  # tie-breaking

    def update(self, arm, reward):
        self.t += 1
        if self.context_collection is None:
            self.context_collection = self.context[arm]
        else:
            self.context_collection = np.vstack((self.context_collection, self.context[arm]))  # vertical stacking
        self.rewards = np.append(self.rewards, reward)
        if self.rewards.size >= 2:
            self.init = False
            self.K_inv = np.linalg.inv(self.kernel(self.context_collection))

    def add_arm(self):
        self.num_arms += 1

    def remove_arm(self, arm):
        self.num_arms -= 1

    def __str__(self):
        return 'GP-UCB'

    def __update_beta(self):
        if not self.beta_fixed:
            self.beta = (self.num_dimensions * 2 + 2) * np.log(self.t * self.hyper_const)

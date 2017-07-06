import abc
import numpy as np
import sys


class BanditAlgorithm(object):
    """
    An interface of bandit algorithms, which has three primitive methods: select_arm, update, reset
    Refer to: John Myles White. Bandit Algorithms for Website Optimization. O'Reilly Media, Inc., 2012.
    """
    __metaclass__ = abc.ABCMeta

    """
    Throughout codes, all of the bandit algorithms will implement this select_arm method,
    that is called without any arguments and which returns the index of the next arm to pull.
    - Return: the index of the arm within [0, the number of arms - 1], which gives the highest expectation
    """
    @abc.abstractmethod
    def select_arm(self):
        pass

    """
    After we pull an arm, we get a reward signal back from our system.
    It updates the algorithm's beliefs about the quality of the arm we just chose, by providing the reward information.
    - arm: the arm just pulled at the moment
    - reward: the reward received from choosing the arm
    """
    @abc.abstractmethod
    def update(self, arm, reward):
        pass

    """
    Initialize all instance variables declared in the class.
    """
    @abc.abstractmethod
    def reset(self):
        pass

    """
    (Additional method)
    Add an arm for the algorithm to take into account.
    - prior: the prior knowledge (or information) of the arm, if any
    """
    @abc.abstractmethod
    def add_arm(self, prior):
        pass

    """
    (Additional method)
    Remove an arm for the algorithm not to take into account.
    - arm: the arm needs to be removed
    """
    @abc.abstractmethod
    def remove_arm(self, arm):
        pass


class AverageBanditAlgorithm(BanditAlgorithm):
    """
    BanditAlgorithm that utilizes the mean of the rewards.
    The algorithm always chooses the arm with the highest estimated mean, based on the rewards observed thus far.
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.averages = np.zeros(num_arms)

    def select_arm(self):
        # tie-breaking, https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking
        return np.random.choice(np.flatnonzero(self.averages == self.averages.max()))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.__update_average(arm, reward)

    def reset(self):
        self.counts.fill(0)
        self.averages.fill(0)

    def add_arm(self, prior=None):
        # prior:
        # (case 1) single value, the average of the adding arm
        # (case 2) (average, count) of the adding arm (numpy array or python list)
        self.num_arms += 1
        self.counts = np.append(self.counts, np.zeros(1))  # add one zero
        if prior is None:
            add_value = 0
        else:
            if hasattr(prior, "__len__"):  # numpy array and list have __len__ attribute
                add_value = prior[0]
                self.counts[self.num_arms - 1] = prior[1]
            else:
                add_value = prior
        self.averages = np.append(self.averages, add_value)

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.counts = np.delete(self.counts, arm)
        self.averages = np.delete(self.averages, arm)

    def __str__(self):
        return 'Average'

    def __update_average(self, arm, reward):
        counts_arm = float(self.counts[arm])
        self.averages[arm] = ((counts_arm - 1) / counts_arm) * self.averages[arm] + (1 / counts_arm) * reward


class EpsilonFirstAlgorithm(AverageBanditAlgorithm):
    """
    BanditAlgorithm that chooses a random arm during the first epsilon draws
    (exploration phase), and the arm with the highest estimated mean after that (exploitation phase).
    http://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """
    def __init__(self, num_arms, epsilon, reset_when_arm_changes=True):
        super(EpsilonFirstAlgorithm, self).__init__(num_arms)
        self.current_draw = 0
        self.epsilon = epsilon
        self.reset_when_arm_changes = reset_when_arm_changes

    def select_arm(self):
        self.current_draw += 1
        if self.current_draw <= self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return super(EpsilonFirstAlgorithm, self).select_arm()

    def reset(self):
        super(EpsilonFirstAlgorithm, self).reset()
        self.current_draw = 0

    def add_arm(self, prior=None):
        super(EpsilonFirstAlgorithm, self).add_arm(prior)
        if self.reset_when_arm_changes:
            self.reset()

    def remove_arm(self, arm):
        super(EpsilonFirstAlgorithm, self).remove_arm(arm)
        if self.reset_when_arm_changes:
            self.reset()

    def __str__(self):
        return 'EpsilonFirst(epsilon={})'.format(self.epsilon)


class EpsilonGreedyAlgorithm(AverageBanditAlgorithm):
    """
    BanditAlgorithm that chooses a random arm with epsilon-frequency,
    and otherwise chooses the arm with the highest estimated mean, based on the rewards observed thus far.
    http://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """
    def __init__(self, num_arms, epsilon):
        super(EpsilonGreedyAlgorithm, self).__init__(num_arms)
        self.epsilon = epsilon

    def select_arm(self):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return super(EpsilonGreedyAlgorithm, self).select_arm()

    def __str__(self):
        if self.epsilon == 1:
            return 'Random'
        else:
            return 'EpsilonGreedy(epsilon={})'.format(self.epsilon)


class SoftmaxAlgorithm(AverageBanditAlgorithm):
    """
    BanditAlgorithm that chooses the best arm, by considering proportions of their estimated reward rates.
    http://duoduo2011.blogspot.kr/2016/03/5-softmax-algorithm.html
    """
    def __init__(self, num_arms, temperature):
        super(SoftmaxAlgorithm, self).__init__(num_arms)
        self.temperature = temperature

    def select_arm(self):
        numerator = np.exp(self.averages / self.temperature)
        probabilities = numerator / numerator.sum()
        return np.random.choice(self.num_arms, p=probabilities)

    def __str__(self):
        return 'Softmax(temperature={})'.format(self.temperature)


class UCB1Algorithm(AverageBanditAlgorithm):
    """
    BanditAlgorithm that is very popular in the domain of multi-armed bandit.
    - The use of Upper Confidence Bounds(aka UCB1) automatically trades off between exploitation and exploration.
    - UCB1 doesn't have any free parameters to configure before deploying it.
    - UCB1 finds the best arm very quickly,
      but the backpedaling it does causes it to underperform the Softmax algorithm along most metrics.
    Refer to: Peter Auer. "Using confidence bounds for exploitation-exploration trade-offs."
              Journal of Machine Learning Research 3.Nov (2002): 397-422.
    """
    def select_arm(self):
        zero_ind = np.where(self.counts == 0)[0]
        if len(zero_ind):
            return np.random.choice(zero_ind)
        ucb_values = self.averages + np.sqrt(2 * np.log(self.counts.sum()) / self.counts)
        return np.random.choice(np.flatnonzero(ucb_values == ucb_values.max()))

    def __str__(self):
        return 'UCB1'


class UCBTunedAlgorithm(UCB1Algorithm):
    """
    BanditAlgorithm that improves over the solution of Ucb1Algorithm,
    and this can be made by replacing the second term(sqrt of 2*log(t)/n_i) of UCB1 with the tuned term.
    - UCB-Tuned is not very sensitive to the variance of the arms.
    - UCB-Tuned outperforms the earlier UCBs(UCB1, UCB2) significantly in experiments.
    Refer to: Burtini et al. "A Survey of Online Experiment Design with the Stochastic Multi-Armed Bandit."
              arXiv preprint arXiv:1510.00757 (2015).
    """
    def __init__(self, num_arms):
        super(UCBTunedAlgorithm, self).__init__(num_arms)
        self.average_of_squares = np.zeros(num_arms)

    def select_arm(self):
        zero_ind = np.where(self.counts == 0)[0]
        if len(zero_ind):
            return np.random.choice(zero_ind)
        variance = self.average_of_squares - np.power(self.averages, 2)
        log_t_div_cnt = np.log(self.counts.sum()) / self.counts
        ucb_values = self.averages + np.sqrt(log_t_div_cnt * np.minimum(0.25, variance + np.sqrt(2 * log_t_div_cnt)))
        return np.random.choice(np.flatnonzero(ucb_values == ucb_values.max()))

    def update(self, arm, reward):
        super(UCBTunedAlgorithm, self).update(arm, reward)
        self.__update_average_of_square(arm, reward)

    def reset(self):
        super(UCBTunedAlgorithm, self).reset()
        self.average_of_squares.fill(0)

    def add_arm(self, prior=None):
        # prior: the average of the adding arm
        super(UCBTunedAlgorithm, self).add_arm(prior)
        if prior is None:
            add_value = 0
        else:
            if hasattr(prior, "__len__"):
                add_value = prior[0] * prior[0]
            else:
                add_value = prior * prior
        self.average_of_squares = np.append(self.average_of_squares, add_value)

    def remove_arm(self, arm):
        super(UCBTunedAlgorithm, self).remove_arm(arm)
        self.average_of_squares = np.delete(self.average_of_squares, arm)

    def __str__(self):
        return 'UCB-Tuned'

    def __update_average_of_square(self, arm, reward):
        counts_arm = float(self.counts[arm])
        self.average_of_squares[arm] = ((counts_arm - 1) / counts_arm) * self.average_of_squares[arm] +\
                                       (1 / counts_arm) * reward * reward


class UCBVAlgorithm(UCBTunedAlgorithm):
    """
    BanditAlgorithm that improves over the solution of Ucb1Algorithm, by using variance estimates in the bias sequence.
    - The algorithm described in the original paper should takes two input parameters named as b and c.
      However the parameters are multiplied together in the calculation of the confidence bounds.
    - Therefore you must select the parameter properly for practical usages.
    Refer to: Audibert et al. "Tuning bandit algorithms in stochastic environments."
              International conference on Algorithmic Learning Theory. Springer Berlin Heidelberg, 2007.
    """
    def __init__(self, num_arms, parameter):
        super(UCBVAlgorithm, self).__init__(num_arms)
        self.parameter = parameter

    def select_arm(self):
        zero_ind = np.where(self.counts == 0)[0]
        if len(zero_ind):
            return np.random.choice(zero_ind)
        variance = self.average_of_squares - np.power(self.averages, 2)
        log_total = np.log(self.counts.sum())
        bonus = np.sqrt(2 * log_total * variance / self.counts) + 3 * self.parameter * log_total / self.counts
        ucb_values = self.averages + bonus
        return np.random.choice(np.flatnonzero(ucb_values == ucb_values.max()))

    def __str__(self):
        return 'UCB-V(parameter={})'.format(self.parameter)


class BayesBanditAlgorithm(BanditAlgorithm):
    """
    BanditAlgorithm that is a heuristic for choosing actions to maximize the expected rewards
    (aka Thompson sampling, Bayesian bandit) using the Bayesian framework.
    - The algorithm keeps a beta-distribution for each arm and update it according to the trials and successes
      (and failures) you have seen so far, and takes a sample from each distribution and choose the arm.
    - The algorithm needs the specification of the prior probability for inference. Default is prior=1.
      (Bayes' prior, see in https://en.wikipedia.org/wiki/Beta_distribution#Bayesian_inference for details)
    - The algorithm only supports arms whose rewards are sampled from the underlying Bernoulli and
      binomial distributions. The Beta distribution is conjugate to the Bernoulli and the binomial.
      (See in https://en.wikipedia.org/wiki/Conjugate_prior for details)
    Refer to: Chapelle and Li. "An empirical evaluation of thompson sampling."
              Advances in neural information processing systems. 2011.
    """
    def __init__(self, num_arms, num_priors=1.0):
        self.num_arms = num_arms
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
        self.num_priors = num_priors

    def select_arm(self):
        a = self.successes + self.num_priors
        b = self.failures + self.num_priors
        samples = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            samples[i] = np.random.beta(a[i], b[i])
        return np.random.choice(np.flatnonzero(samples == samples.max()))

    def update(self, arm, reward):
        self.successes[arm] += reward
        self.failures[arm] += self.num_priors - reward

    def reset(self):
        self.successes.fill(0)
        self.failures.fill(0)

    def add_arm(self, prior=None):
        # prior: the number of successes and failures of the adding arm (numpy array or python list)
        self.num_arms += 1
        if prior is None:
            add_value1 = 0
            add_value2 = 0
        else:
            add_value1 = prior[0]
            add_value2 = prior[1]
        self.successes = np.append(self.successes, add_value1)
        self.failures = np.append(self.failures, add_value2)

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.successes = np.delete(self.successes, arm)
        self.failures = np.delete(self.failures, arm)

    def __str__(self):
        if self.num_priors == 1.0:
            return 'Bayes'
        else:
            return 'Bayes(prior={})'.format(self.num_priors)


class Exp3SAlgorithm(BanditAlgorithm):
    """
    Adversarial Bandit Algorithm - Exp3.S
    Exp3 stands for "Exponential-weight algorithm for Exploration and Exploitation".
    Peter Auer et al. "The nonstochastic multiarmed bandit problem." SIAM journal on computing 32.1 (2002): 48-77.
    - Requirements: gamma in (0,1], alpha > 0 (equivalent to Exp3 when alpha == 0)
    """
    def __init__(self, num_arms, gamma, alpha=0.0):
        self.num_arms = num_arms
        self.gamma = gamma
        self.alpha = alpha
        self.weights = np.ones(num_arms)
        self.probabilities = np.ones(num_arms) / num_arms

    def select_arm(self):
        self.__update_probabilities()
        return np.random.choice(self.num_arms, p=self.probabilities)

    def update(self, arm, reward):
        self.weights *= np.exp(self.gamma * self.__get_estimated_reward(arm, reward) / self.num_arms)
        self.weights += np.e * self.alpha / self.num_arms * self.weights.sum()

        # optional but maybe essential: prevents overflow
        self.weights += sys.float_info.min
        self.weights /= self.weights.sum()

    def reset(self, gamma=None, alpha=None):
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha
        self.weights.fill(1.0)
        self.probabilities.fill(1.0 / self.num_arms)

    def add_arm(self, prior=None):
        # prior: the weight of the adding arm, in [0,1]
        self.num_arms += 1
        if prior is None:
            self.weights = np.append(self.weights, np.zeros(1))
        else:
            self.weights = np.append(self.weights, prior)
            self.weights /= self.weights.sum()
        self.__update_probabilities()

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.weights = np.delete(self.weights, arm)
        self.__update_probabilities()

    def __str__(self):
        if self.alpha == 0.0:
            return 'Exp3(gamma={})'.format(self.gamma)
        else:
            return 'Exp3.S(gamma={},alpha={})'.format(self.gamma, self.alpha)

    def __get_estimated_reward(self, arm, reward):
        estimated_reward = np.zeros(self.num_arms)
        estimated_reward[arm] = reward / self.probabilities[arm]
        return estimated_reward

    def __update_probabilities(self):
        self.probabilities = (1.0 - self.gamma) * self.weights / self.weights.sum() + self.gamma / self.num_arms


class Exp3Algorithm(Exp3SAlgorithm):
    """
    Adversarial Bandit Algorithm - Exp3
    Peter Auer et al. "The nonstochastic multiarmed bandit problem." SIAM journal on computing 32.1 (2002): 48-77.
    - Requirement: gamma in (0,1]
    """
    def __init__(self, num_arms, gamma):
        super(Exp3Algorithm, self).__init__(num_arms, gamma)


class Exp31Algorithm(BanditAlgorithm):
    """
    Adversarial Bandit Algorithm - Exp3.1
    Peter Auer et al. "The nonstochastic multiarmed bandit problem." SIAM journal on computing 32.1 (2002): 48-77.
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.epoch = 0  # r = 0, 1, 2, ...
        self.g = 0  # upper bound on the total estimated reward
        self.__update_g()
        self.gamma = 1.0  # gamma starts at 1.0 for epoch=0
        self.total = np.zeros(num_arms)  # total estimated reward
        self.Exp3S = Exp3Algorithm(num_arms, self.gamma)

    def select_arm(self):
        if self.total.max() > self.g - self.num_arms / self.gamma:
            self.epoch += 1
            self.g *= 4
            self.gamma *= 0.5
            self.Exp3S.reset(self.gamma)
        return self.Exp3S.select_arm()

    def update(self, arm, reward):
        self.Exp3S.update(arm, reward)
        self.total[arm] += reward / self.Exp3S.probabilities[arm]

    def reset(self):
        self.epoch = 0
        self.__update_g()
        self.gamma = 1.0
        self.total.fill(0.0)
        self.Exp3S.reset(self.gamma)

    def add_arm(self, prior=None):
        # prior: the weight of the adding arm, in [0,1]
        self.num_arms += 1
        self.__update_g()
        self.total = np.append(self.total, np.zeros(1))
        self.Exp3S.add_arm(prior)

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.__update_g()
        self.total = np.delete(self.total, arm)
        self.Exp3S.remove_arm(arm)

    def __str__(self):
        return 'EXP3.1'

    def __update_g(self):
        self.g = self.num_arms * np.log(self.num_arms) / (np.e - 1.0) * np.power(4, self.epoch)


class Exp3S1Algorithm(BanditAlgorithm):
    """
    Adversarial Bandit Algorithm - Exp3.S.1
    Peter Auer et al. "The nonstochastic multiarmed bandit problem." SIAM journal on computing 32.1 (2002): 48-77.
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.epoch = 0
        self.num_draws = 1
        self.alpha = 1.0
        self.prev_selected_arm = None
        self.hardness = 1
        self.gamma = 0.0
        self.__update_gamma()
        self.Exp3S = Exp3SAlgorithm(num_arms, self.gamma, self.alpha)
        self.t = 1  # current time count
        self.next_t = self.num_draws  # time limit, until the current Exp3S is valid
        self.selected_history = []  # empty list

    def select_arm(self):
        selected_arm = self.Exp3S.select_arm()
        if self.next_t == self.t:
            self.epoch += 1
            self.num_draws *= 2
            self.next_t += self.num_draws
            self.alpha = 1.0 / self.num_draws
            self.hardness = 1
            self.__update_gamma()
            self.Exp3S.reset(self.gamma, self.alpha)
        return selected_arm

    def update(self, arm, reward):
        self.t += 1
        self.Exp3S.update(arm, reward)
        self.selected_history.append(arm)
        if arm != self.prev_selected_arm:
            self.hardness += 1
            self.prev_selected_arm = arm

    def reset(self):
        self.epoch = 0
        self.num_draws = 1
        self.alpha = 1.0
        self.prev_selected_arm = None
        self.hardness = 1
        self.__update_gamma()
        self.Exp3S.reset(self.gamma, self.alpha)
        self.t = 0
        self.next_t = self.num_draws
        self.selected_history.clear()

    def add_arm(self, prior=None):
        self.num_arms += 1
        self.__update_gamma()
        self.Exp3S.add_arm(prior)

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.selected_history =\
            list(filter(lambda a: a != arm, self.selected_history))  # remove the pulled occurrences of the arm
        self.__update_hardness()  # re-calculate hardness
        self.__update_gamma()
        self.Exp3S.remove_arm(arm)

    def __str__(self):
        return 'Exp3.S.1'

    def __update_gamma(self):
        root_inner = self.num_arms * (self.hardness * np.log(self.num_arms * self.num_draws) + np.e) /\
                     ((np.e - 1) * self.num_draws)
        self.gamma = np.minimum(1.0, np.sqrt(root_inner))

    def __update_hardness(self):
        len_history = len(self.selected_history)
        diff = np.array(self.selected_history[:len_history - 1]) - np.array(self.selected_history[1:len_history])
        self.hardness = (diff < 0).sum() + 1


class Exp3PAlgorithm(BanditAlgorithm):
    """
    Adversarial Bandit Algorithm - Exp3.P
    Peter Auer et al. "The nonstochastic multiarmed bandit problem." SIAM journal on computing 32.1 (2002): 48-77.
    - Requirements: gamma in (0,1], alpha > 0
    """
    def __init__(self, num_arms, gamma, alpha, num_draws):
        self.num_arms = num_arms
        self.gamma = gamma
        self.alpha = alpha
        self.num_draws = num_draws
        self.probabilities = np.ones(num_arms) / num_arms
        self.weights = np.ones(num_arms) * np.exp(alpha * self.gamma / 3 * np.sqrt(num_draws / num_arms))

    def select_arm(self):
        self.__update_probabilities()
        return np.random.choice(self.num_arms, p=self.probabilities)

    def update(self, arm, reward):
        inner = self.__get_estimated_reward(arm, reward) +\
                self.alpha / (self.probabilities * np.sqrt(self.num_arms * self.num_draws))
        self.weights *= np.exp(self.gamma / (3 * self.num_arms) * inner)

        # optional but maybe essential: prevents overflow
        self.weights += sys.float_info.min
        self.weights /= self.weights.sum()

    def reset(self, gamma=None, alpha=None, num_draws=None):
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha
        if num_draws is not None:
            self.num_draws = num_draws
        self.probabilities.fill(1.0 / self.num_arms)
        self.weights = np.ones(self.num_arms) *\
            np.exp(self.alpha * self.gamma / 3 * np.sqrt(self.num_draws / self.num_arms))

    def add_arm(self, prior=None):
        # prior: the weight of the adding arm, in [0,1]
        self.num_arms += 1
        if prior is None:
            self.weights = np.append(self.weights, np.zeros(1))
        else:
            self.weights = np.append(self.weights, prior)
            self.weights /= self.weights.sum()
        self.__update_probabilities()

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.weights = np.delete(self.weights, arm)
        self.__update_probabilities()

    def __str__(self):
        return 'Exp3.P(gamma={},alpha={},T={})'.format(self.gamma, self.alpha, self.num_draws)

    def __get_estimated_reward(self, arm, reward):
        estimated_reward = np.zeros(self.num_arms)
        estimated_reward[arm] = reward / self.probabilities[arm]
        return estimated_reward

    def __update_probabilities(self):
        self.probabilities = (1.0 - self.gamma) * self.weights / self.weights.sum() + self.gamma / self.num_arms


class Exp3P1Algorithm(BanditAlgorithm):
    """
    Adversarial Bandit Algorithm - Exp3.P.1
    Peter Auer et al. "The nonstochastic multiarmed bandit problem." SIAM journal on computing 32.1 (2002): 48-77.
    - Requirement: delta in (0,1)
    """
    def __init__(self, num_arms, delta, num_draws=1):
        self.num_arms = num_arms
        self.delta = delta
        # restrict the value to the power of 2 and to be nearest to the original num_draws
        self.num_draws = np.power(2, np.log(num_draws).round())
        self.r = np.log2(self.num_draws)
        self.delta_r = 0.0
        self.r_star = 0
        self.__init_r_values()
        self.__update_gamma_and_alpha()
        self.Exp3P = Exp3PAlgorithm(num_arms, self.gamma, self.alpha, self.num_draws)
        self.t = 1
        self.next_t = self.num_draws

    def select_arm(self):
        selected_arm = self.Exp3P.select_arm()
        if self.next_t == self.t:
            self.r += 1
            self.num_draws *= 2
            self.next_t += self.num_draws
            self.delta_r = self.delta / ((self.r + 1) * (self.r + 2))
            self.__update_gamma_and_alpha()
            self.Exp3P.reset(self.gamma, self.alpha, self.num_draws)
        return selected_arm

    def update(self, arm, reward):
        self.t += 1
        self.Exp3P.update(arm, reward)

    def reset(self):
        self.r = self.r_star
        self.num_draws = np.power(2, self.r)
        self.delta_r = self.delta / ((self.r + 1) * (self.r + 2))
        self.__update_gamma_and_alpha()
        self.Exp3P.reset(self.gamma, self.alpha, self.num_draws)
        self.t = 0
        self.next_t = self.num_draws

    def add_arm(self, prior=None):
        self.num_arms += 1
        self.Exp3P.add_arm(prior)

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.Exp3P.remove_arm(arm)

    def __str__(self):
        return 'Exp3.P.1(delta={})'.format(self.delta)

    def __init_r_values(self):
        flag_smaller = False
        while True:
            r = np.log2(self.num_draws)
            delta_r = self.delta / ((r + 1) * (r + 2))
            kt = self.num_arms * self.num_draws
            if delta_r >= kt * np.exp(-kt):
                if self.num_draws == 1:  # possible minimum number of draws (r=0)
                    break
                if flag_smaller is False:
                    self.num_draws /= 2
                else:
                    break
            else:
                self.num_draws *= 2
                flag_smaller = True
        self.r = self.r_star = r
        self.delta_r = delta_r

    def __update_gamma_and_alpha(self):
        self.gamma = np.minimum(0.6, 2 * np.sqrt(0.6 * self.num_arms * np.log(self.num_arms) / self.num_draws))
        self.alpha = 2 * np.sqrt(np.log(self.num_arms * self.num_draws / self.delta_r))


class Exp4Algorithm(object):
    """
    Adversarial Bandit Algorithm - Exp4
    Exp4 stands for "Exponential-weight algorithm for Exploration and Exploitation using Expert advice".
    Peter Auer et al. "The nonstochastic multiarmed bandit problem." SIAM journal on computing 32.1 (2002): 48-77.
    - Requirement: gamma in (0,1]
    """
    def __init__(self, num_arms, num_experts, gamma):
        self.num_arms = num_arms
        self.num_experts = num_experts
        self.gamma = gamma
        self.weights = np.ones(num_experts)
        self.probabilities = np.ones(num_arms) / num_arms
        self.advice = None

    def select_arm(self, advice):
        # weights: 1 x N array, advice: N x K matrix, prob: 1 x K array (N: num_experts, K: num_arms)
        self.advice = advice
        if self.num_experts == 1:
            w_dot_xi = self.weights * advice
        else:
            w_dot_xi = self.weights.dot(advice)
        self.probabilities = (1.0 - self.gamma) * w_dot_xi / self.weights.sum() + self.gamma / self.num_arms
        return np.random.choice(self.num_arms, p=self.probabilities)

    def update(self, arm, reward):
        self.weights *= np.exp(self.gamma * self.__get_estimated_y(arm, reward) / self.num_arms)

        # optional but maybe essential: prevents overflow
        self.weights += sys.float_info.min
        self.weights /= self.weights.sum()

    def reset(self, gamma=None):
        if gamma is not None:
            self.gamma = gamma
        self.weights.fill(1.0)
        self.probabilities.fill(1.0 / self.num_arms)

    def add_arm(self):
        self.num_arms += 1
        self.probabilities = np.append(self.probabilities, np.zeros(1))
        if self.advice is not None:
            if self.num_experts == 1:
                self.advice = np.append(self.advice, np.zeros(1))
            else:
                self.advice = np.hstack((self.advice, np.zeros((self.num_experts, 1))))

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.probabilities = np.delete(self.probabilities, arm)
        if self.advice is not None:
            if self.num_experts == 1:
                self.advice = np.delete(self.advice, arm)
            else:
                self.advice = np.delete(self.advice, arm, axis=1)

    def __str__(self):
        return 'Exp4(gamma={})'.format(self.gamma)

    def __get_estimated_y(self, arm, reward):
        estimated_reward = np.zeros(self.num_arms)
        estimated_reward[arm] = reward / self.probabilities[arm]  # to here, same as Exp3
        return self.advice.dot(estimated_reward)  # but this line is different


class Exp4PAlgorithm(object):
    """
    Adversarial Bandit Algorithm - Exp4.P
    Alina Beygelzimer et al. "Contextual bandit algorithms with supervised learning guarantees".
    Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. 2011.
    - Requirement: delta > 0
    """
    def __init__(self, num_arms, num_experts, delta, num_draws, tighter_bounds=True):
        self.num_arms = num_arms
        self.num_experts = num_experts
        self.delta = delta
        self.num_draws = num_draws
        self.weights = np.ones(num_experts)
        self.probabilities = np.ones(num_arms) / num_arms
        self.advice = None
        self.p_min = 0
        self.__update_p_min()
        self.tighter_bounds = tighter_bounds

    def select_arm(self, advice):
        # weights: 1 x N numpy array, advice: N x K numpy matrix, p: 1 x K numpy array
        # (N: num_experts, K: num_arms)
        self.advice = advice
        if self.num_experts == 1:
            p = self.weights * advice
        else:
            p = self.weights.dot(advice)
        p /= self.weights.sum()

        # An alternative method for setting probabilities:
        # McMahan and Streeter. "Tighter Bounds for Multi-Armed Bandits with Expert Advice". COLT. 2009. 
        if self.tighter_bounds:  # improved method, in chapter 6 (algorithm 2)
            cap_delta = 0.0
            l = 1.0
            i = 0
            ind_asc = np.argsort(p)
            for j in np.nditer(ind_asc):
                m = 1 - cap_delta / l
                if p[j] * m >= self.p_min:
                    ind_update = ind_asc[i:]
                    self.probabilities[ind_update] = p[ind_update] * m
                    break
                else:
                    self.probabilities[j] = self.p_min
                    cap_delta += self.p_min - p[j]
                    l -= p[j]
                    i += 1
        else:  # original, in chapter 4 (algorithm 1)
            self.probabilities = (1.0 - self.num_arms * self.p_min) * p + self.p_min

        return np.random.choice(self.num_arms, p=self.probabilities)

    def update(self, arm, reward):
        y, v = self.__get_estimated_y_v(arm, reward)
        inner = y + v * np.sqrt(np.log(self.num_experts / self.delta) / (self.num_arms * self.num_draws))
        self.weights *= np.exp(self.p_min * 0.5 * inner)

        # optional but maybe essential: prevents overflow
        self.weights += sys.float_info.min
        self.weights /= self.weights.sum()

    def reset(self):
        self.weights.fill(1.0)
        self.probabilities.fill(1.0 / self.num_arms)

    def add_arm(self):
        self.num_arms += 1
        self.probabilities = np.append(self.probabilities, np.zeros(1))
        self.__update_p_min()
        if self.advice is not None:
            if self.num_experts == 1:
                self.advice = np.append(self.advice, np.zeros(1))
            else:
                self.advice = np.hstack((self.advice, np.zeros((self.num_experts, 1))))

    def remove_arm(self, arm):
        self.num_arms -= 1
        self.probabilities = np.delete(self.probabilities, arm)
        self.__update_p_min()
        if self.advice is not None:
            if self.num_experts == 1:
                self.advice = np.delete(self.advice, arm)
            else:
                self.advice = np.delete(self.advice, arm, axis=1)

    def __str__(self):
        return 'Exp4.P(delta={},T={})'.format(self.delta, self.num_draws)

    def __get_estimated_y_v(self, arm, reward):
        estimated_reward = np.zeros(self.num_arms)
        estimated_reward[arm] = reward / self.probabilities[arm]
        return self.advice.dot(estimated_reward), self.advice.dot(1 / self.probabilities)

    def __update_p_min(self):
        self.p_min = np.sqrt(np.log(self.num_experts) / (self.num_arms * self.num_draws))

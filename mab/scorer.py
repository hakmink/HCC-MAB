import abc
import numpy as np


class Scorer(object):
    """
    An interface that is used to aggregate scores, for each algorithm.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_score(self, draw, chosen_arm, reward):
        pass


class AverageRewardScorer(Scorer):
    """
    Scorer that averages the rewards.
    """
    def __init__(self):
        self.accumulated_score = 0.0

    def update_score(self, draw, chosen_arm, reward):
        self.accumulated_score += reward
        return self.accumulated_score / draw


class BestArmSelectedScorer(Scorer):
    """
    Scorer that calculates the selection ratio of the best arm, which has the greatest expectation of reward.
    """
    def __init__(self, arms):
        self.arms = arms
        self.selected_best_count = 0

    def update_score(self, draw, chosen_arm, reward):
        num_arms = len(self.arms)
        expected_values = np.zeros(num_arms)
        for i in range(num_arms):
            expected_values[i] = self.arms[i].get_expected_value()
        max_arms = np.flatnonzero(expected_values == expected_values.max())
        if chosen_arm in max_arms:
            self.selected_best_count += 1
        return self.selected_best_count / float(draw)


class CumulativeRewardScorer(Scorer):
    """
    Scorer that accumulates the rewards.
    """
    def __init__(self):
        self.accumulated_score = 0.0

    def update_score(self, draw, chosen_arm, reward):
        self.accumulated_score += reward
        return self.accumulated_score

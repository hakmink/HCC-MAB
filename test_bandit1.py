# simple tests of all multi-armed bandit algorithms

from mab import algorithm as bd
from mab import arm
from mab import scorer as sc
import time

num_draws = 4000
print('total number of draws: {}'.format(num_draws))

arms = [
    arm.BernoulliArm(0.8),
    arm.BernoulliArm(0.6),
    arm.BernoulliArm(0.25)
]
num_arms = len(arms)
print('number of arms: {}'.format(num_arms))

algorithms = [
    bd.EpsilonGreedyAlgorithm(num_arms, 0),
    bd.EpsilonGreedyAlgorithm(num_arms, 0.1),
    bd.EpsilonFirstAlgorithm(num_arms, 1000),
    bd.SoftmaxAlgorithm(num_arms, 0.1),
    bd.UCB1Algorithm(num_arms),
    bd.UCBTunedAlgorithm(num_arms),
    bd.UCBVAlgorithm(num_arms, 0.1),
    bd.BayesBanditAlgorithm(num_arms),
    bd.Exp3Algorithm(num_arms, 0.1),
    bd.Exp31Algorithm(num_arms),
    bd.Exp3SAlgorithm(num_arms, 0.1, 0.002),
    bd.Exp3S1Algorithm(num_arms),
    bd.Exp3PAlgorithm(num_arms, 0.1, 0.1, num_draws),
    bd.Exp3P1Algorithm(num_arms, 0.1)
]
num_algorithms = len(algorithms)

for i in range(num_algorithms):
    scorers = [
        sc.AverageRewardScorer(),
        sc.BestArmSelectedScorer(arms),
        sc.CumulativeRewardScorer()
    ]

    elapsed_time = 0.0
    avg_score, best_score, cum_score = 0.0, 0.0, 0.0

    for j in range(num_draws):
        start_time = time.perf_counter()
        selected_arm = algorithms[i].select_arm()
        reward = arms[selected_arm].draw()
        algorithms[i].update(selected_arm, reward)
        end_time = time.perf_counter()
        elapsed_time += end_time - start_time

        draw = j + 1

        if j < num_draws - 1:
            for scorer in scorers:
                scorer.update_score(draw, selected_arm, reward)
        else:
            avg_score = scorers[0].update_score(draw, selected_arm, reward)
            best_score = scorers[1].update_score(draw, selected_arm, reward)
            cum_score = scorers[2].update_score(draw, selected_arm, reward)

    elapsed_time *= 1000000
    avg_elapsed_time = elapsed_time / float(num_draws)

    print('algorithm {}: {}\navg_reward: {}, best_selected: {}, cum_reward: {}, total_time: {} microseconds, '
          'avg_time: {} microseconds'.format(i + 1, str(algorithms[i]), avg_score, best_score, cum_score,
                                             elapsed_time, avg_elapsed_time))

# simple tests on dynamic environment (addition/deletion of arms)

from mab import algorithm as bd
from mab import arm
from mab import scorer as sc
import time

num_draws = 10000
print('total number of draws: ', num_draws)

init_arms = [
    arm.BernoulliArm(0.4),
    arm.BernoulliArm(0.3),
    arm.BernoulliArm(0.2),
    arm.BernoulliArm(0.1),
    arm.BernoulliArm(0.25),
    arm.BernoulliArm(0.35)
]
num_arms = len(init_arms)
print('initial number of arms: ', num_arms)

algorithms = [
    bd.EpsilonGreedyAlgorithm(num_arms, 1),
    bd.AverageBanditAlgorithm(num_arms),
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
    scorer = sc.CumulativeRewardScorer()

    elapsed_time = 0.0
    avg_score, best_score, cum_score = 0.0, 0.0, 0.0
    max_expected_cum_reward = 0.0

    arms = init_arms[:]  # copy arms by value, not by reference

    for j in range(num_draws):
        # event 1: arm addition
        if j == num_draws * 0.25:
            algorithms[i].add_arm()
            arms.append(arm.BernoulliArm(0.45))

        # event 2: arm deletion
        if j == num_draws * 0.4:
            algorithms[i].remove_arm(1)
            del arms[1]

        # event 3: arm addition
        if j == num_draws * 0.5:
            algorithms[i].add_arm()
            arms.append(arm.BernoulliArm(0.325))

        # event 4: arm deletion
        if j == num_draws * 0.65:
            algorithms[i].remove_arm(4)
            del arms[4]

        """
        if j == num_draws * 0.75:
            algorithms[i].remove_arm(1)
            del arms[1]

        if j == num_draws * 0.85:
            algorithms[i].remove_arm(1)
            del arms[1]

        if j == num_draws * 0.95:
            algorithms[i].remove_arm(1)
            del arms[1]
        """

        addition = max([arm.get_expected_value() for arm in arms])
        max_expected_cum_reward += addition

        start_time = time.perf_counter()
        selected_arm = algorithms[i].select_arm()
        reward = arms[selected_arm].draw()

        algorithms[i].update(selected_arm, reward)
        end_time = time.perf_counter()
        elapsed_time += end_time - start_time

        draw = j + 1

        if j < num_draws - 1:
            scorer.update_score(draw, selected_arm, reward)
        else:
            cum_score = scorer.update_score(draw, selected_arm, reward)

    elapsed_time *= 1000000
    avg_elapsed_time = elapsed_time / float(num_draws)

    print('algorithm {}: {}\ncum_reward: {}/{}, total_time: {} microseconds, avg_time: {} microseconds'
          .format(i + 1, str(algorithms[i]), cum_score, int(max_expected_cum_reward), elapsed_time, avg_elapsed_time))

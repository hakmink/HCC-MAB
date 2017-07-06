# prints averages each step, for looking at the behavior of the epsilon-greedy algorithm

from mab import algorithm as bd
from mab import arm
from mab import scorer as sc
import time

num_draws = 20
print('total number of draws: {}'.format(num_draws))

arms = [
    arm.BernoulliArm(0.8),
    arm.BernoulliArm(0.6),
    arm.BernoulliArm(0.25)
]
num_arms = len(arms)
print('number of arms: {}'.format(num_arms))

algorithm = bd.EpsilonGreedyAlgorithm(num_arms, 0.1)
print('algorithm: ' + str(algorithm))

scorers = [
    sc.AverageRewardScorer(),
    sc.BestArmSelectedScorer(arms),
    sc.CumulativeRewardScorer()
]

elapsed_time = 0.0
avg_score, best_score, cum_score = 0.0, 0.0, 0.0

for i in range(num_draws):
    start_time = time.perf_counter()
    selected_arm = algorithm.select_arm()
    reward = arms[selected_arm].draw()
    algorithm.update(selected_arm, reward)
    end_time = time.perf_counter()
    elapsed_time += end_time - start_time

    print('iteration {}, selected_arm: {}, reward_of_selected_arm: {}, '
          'reward_averages_of_arms: {}'.format(i + 1, selected_arm, reward, algorithm.averages))

    draw = i + 1

    if i < num_draws - 1:
        for scorer in scorers:
            scorer.update_score(draw, selected_arm, reward)
    else:
        avg_score = scorers[0].update_score(draw, selected_arm, reward)
        best_score = scorers[1].update_score(draw, selected_arm, reward)
        cum_score = scorers[2].update_score(draw, selected_arm, reward)

elapsed_time *= 1000000
avg_elapsed_time = elapsed_time / float(num_draws)

print('avg_reward: {}, best_selected: {}, cum_reward: {}, total_time: {} microseconds, '
      'avg_time: {} microseconds'.format(avg_score, best_score, cum_score, elapsed_time, avg_elapsed_time))

# Exp4 adversarial bandit algorithm example (contextual)

from mab import algorithm as bd
from mab import arm
from mab import scorer as sc
import numpy as np
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

# advice = np.array([0.49, 0.36, 0.15])
# advice = np.array([[0.49, 0.36, 0.15], [0.4, 0.3, 0.3]])
# advice = np.array([[0.49, 0.36, 0.15], [0.4, 0.3, 0.3], [0.9, 0.1, 0.0]])
advice = np.array([[0.49, 0.36, 0.15],  # sum of each row = 1
                   [0.4, 0.3, 0.3],
                   [0.9, 0.1, 0.0],
                   [0.2, 0.3, 0.5]])  # expert 3([0.9, 0.1, 0.0]) recommends best!
print('advice:\n{}'.format(advice))
if advice.ndim == 1:
    num_experts = 1
else:
    num_experts = advice.shape[0]
# algorithm = bd.Exp4Algorithm(num_arms, num_experts, 0.1)
algorithm = bd.Exp4PAlgorithm(num_arms, num_experts, 0.01, num_draws)
print('algorithm: ' + str(algorithm))
print('number of experts: {}'.format(num_experts))

scorers = [
    sc.AverageRewardScorer(),
    sc.BestArmSelectedScorer(arms),
    sc.CumulativeRewardScorer()
]

elapsed_time = 0.0
avg_score, best_score, cum_score = 0.0, 0.0, 0.0

for i in range(num_draws):
    start_time = time.perf_counter()
    selected_arm = algorithm.select_arm(advice)
    reward = arms[selected_arm].draw()
    algorithm.update(selected_arm, reward)
    end_time = time.perf_counter()
    elapsed_time += end_time - start_time

    # weights: the degree of belief in the expert (N experts)
    # probabilities: the estimated confidence for each arm (K arms)
    print('iteration {}, selected_arm: {}, reward_of_selected_arm: {}, weights: {}, '
          'probabilities: {}'.format(i + 1, selected_arm, reward, algorithm.weights, algorithm.probabilities))

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

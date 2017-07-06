# non-contextual mabs for yahoo's front news recommendation

from mab import algorithm as bd
from mab import contextual_algorithm as cbd
from mab import ArticleArms
import numpy as np
import time


num_trials = 200000
print('total number of trials: {}'.format(num_trials))

arms = ArticleArms('yahoo_r6_full.db')
num_arms = arms.get_num_arms()
print('number of arms: {}'.format(num_arms))

algorithms = [
    bd.EpsilonGreedyAlgorithm(num_arms, 1),
    bd.AverageBanditAlgorithm(num_arms),
    bd.EpsilonGreedyAlgorithm(num_arms, 0.1),
    bd.UCBTunedAlgorithm(num_arms),
    bd.UCBVAlgorithm(num_arms, 0.1),
    bd.BayesBanditAlgorithm(num_arms),
    bd.Exp3Algorithm(num_arms, 0.1),
    cbd.LinUCBDisjointAlgorithm(num_arms, 6, 0.2),  # 6: number of article features
    cbd.LinUCBHybridAlgorithm(num_arms, 36, 6, 0.2)  # 36: number of hybrid features
]
num_algorithms = len(algorithms)
print('number of algorithms: {}'.format(num_algorithms))

total_cumulative_rewards = np.zeros(num_algorithms)
trials = np.zeros(num_algorithms)
num_arms_added = 0
removing_arms_indices = []

print_point_divisor = np.power(10, np.ceil(np.log10(num_trials)) - 2)
print_num_trials_divisor = print_point_divisor * 10
newline = True

alg_elapsed_time = [0 for i in range(num_algorithms)]

# replay method (unbiased estimation, Li 2012)
print('evaluation starts at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
start_time = time.perf_counter()

for t in range(1, num_trials + 1):
    arm_event = arms.get_arm_index()
    for i in range(num_algorithms):
        # pool changes dynamically
        if num_arms_added > 0:
            for j in range(num_arms_added):
                algorithms[i].add_arm()
        if bool(removing_arms_indices):
            for j in removing_arms_indices:
                algorithms[i].remove_arm(j)

        # unbiased estimation (algorithm 2)
        alg_start_time = time.perf_counter()
        if isinstance(algorithms[i], cbd.ContextualBanditAlgorithm):  # contextual
            if isinstance(algorithms[i], cbd.LinUCBDisjointAlgorithm):
                arm_algorithm = algorithms[i].select_arm(arms.get_article_features_all())
            if isinstance(algorithms[i], cbd.LinUCBHybridAlgorithm):
                arm_algorithm = algorithms[i].select_arm(arms.get_all_features())
        else:  # non-contextual
            arm_algorithm = algorithms[i].select_arm()

        if arm_algorithm == arm_event:
            reward = float(arms.get_reward(i))
            total_cumulative_rewards[i] += reward
            algorithms[i].update(arm_algorithm, reward)
            trials[i] += 1
            alg_elapsed_time[i] += time.perf_counter() - alg_start_time

    # get the information of the next arms
    num_arms_added, removing_arms_indices = arms.next()

    # display progress
    if t % print_point_divisor == 0:
        if t % print_num_trials_divisor == 0:
            print(t)
            print('- elapsed time: {0:.2f} seconds'.format(time.perf_counter() - start_time))
            formatted_rewards = ['{0:.4f}'.format(i) for i in (total_cumulative_rewards / trials)]
            print('- per-trial reward: {}'.format(formatted_rewards))
            newline = True
        else:
            print('.', end='', flush=True)
            newline = False

if not newline:
    print('')

print('evaluation ends at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))

estimated_rewards = total_cumulative_rewards / trials
for i in range(num_algorithms):
    print('algorithm {0}: {1}, per-trial reward: {2:.6f}, number of trials: {3:.0f}, avg_time: {4:.3f} microseconds'
          .format(i + 1, str(algorithms[i]), estimated_rewards[i], trials[i],
                  alg_elapsed_time[i] / float(trials[i]) * 1000000))
max_index = np.array(estimated_rewards).argmax()
print('winner: {0}, per-trial reward: {1:.6f}'.format(str(algorithms[max_index]), estimated_rewards[max_index]))

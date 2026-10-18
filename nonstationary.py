import numpy as np
import matplotlib.pyplot as plt

def explore_epsilon_greedy(epsilon: float):
    return np.random.binomial(n=1, p=epsilon)

# environment setup
n_arms = 100
rmin = 0
rmax = 100

# reward and simulation setup
initial_arm_rewards = np.random.uniform(rmin, rmax, size=n_arms)
iterations = 1000
epsilon=0.2
mc_sims = 10

exploration_mechanisms = {
    "Epsilon Greedy": lambda: explore_epsilon_greedy(epsilon),
    # "UCB": None,
    # "Thomson Sampling": None
} # all need to have the same lambda signature
drift_settings = np.linspace(0, 1, num=21) * rmax

results_list = []
for i in range(mc_sims):
    print(f"iteration {i}")

    results = {}
    for x in exploration_mechanisms.keys():
        results[x] = {
            "reward_gained": [],
            "reward_possible": [],
            "reward_frac": [],
            "regret": [],
            "explorations": [],
            "explorations_frac": []
        }

    for explore_name, explore_f in exploration_mechanisms.items():
        print(f"\texploration method = {explore_name}")
        for r_variation_potential in sorted(drift_settings):
            print(f"\t\tmax variation = {r_variation_potential}")
            # initial settings setup
            reward_gained = 0
            reward_possible = 0
            explorations = 0
            arm_rewards = initial_arm_rewards
            expected_rewards = [0.0] * n_arms # this should be the same size / shape as arm_rewards
            times_selected = [0] * n_arms # should also be same size/shape as arm_rewards and expected_rewards
            for i in range(iterations):
                # use decided exploration mechanism to choose arm to pull
                explore = explore_f()
                
                if explore:
                    explorations += 1
                    arm = np.random.randint(0, n_arms) # choose which arm to pull at random
                else:
                    arm = np.argmax(expected_rewards)

                # pull arm and get reward
                reward = arm_rewards[arm]
                # update expected reward and times selected
                expected_rewards[arm] = (expected_rewards[arm] * times_selected[arm] + reward) / (times_selected[arm]+1)
                times_selected[arm] += 1

                # update total reward
                reward_gained += reward
                reward_possible += max(arm_rewards)

                # shift actual rewards at random
                for i in range(len(arm_rewards)):
                    arm_rewards[i] = max(min(arm_rewards[i] + np.random.uniform(-r_variation_potential, r_variation_potential), rmax), rmin) # vary, constrained by rmax and rmin

            results[explore_name]["reward_gained"].append(reward_gained)
            results[explore_name]["reward_possible"].append(reward_possible)
            results[explore_name]["reward_frac"].append(reward_gained/reward_possible)
            results[explore_name]["regret"].append(reward_possible - reward_gained)
            results[explore_name]["explorations"].append(explorations)
            results[explore_name]["explorations_frac"].append(explorations/iterations)

    results_list.append(results)

# generate summary data structure (even average over all mc sims)
for explore_name in results.keys():
    for metric in results[explore_name].keys():
        results[explore_name][metric] = np.mean([res[explore_name][metric] for res in results_list], axis=0)

# generate figure for results
fig, ax = plt.subplots()

for explore_name in exploration_mechanisms.keys():
    plt.plot(drift_settings, results[explore_name]["reward_frac"], label=explore_name)

plt.title("Fraction of Total Possible Reward vs. Allowed Drift Range")
plt.xlabel("Maximum Reward Drift Range")
plt.ylabel("Fraction of Total Possible Reward")
plt.legend()

plt.show()

plt.close(fig)
import numpy as np

def explore_epsilon_greedy(epsilon: float):
    return np.random.binomial(n=1, p=epsilon)

n_arms = 100
rmin = 0
rmax = 100

# standard reward setup
arm_rewards = np.random.uniform(rmin, rmax, size=n_arms)

# sparse reward setup
arm_rewards = [0.0] * n_arms
arm_rewards[np.random.randint(0, n_arms)] = 65.0
arm_rewards[np.random.randint(0, n_arms)] = 4.0
arm_rewards[np.random.randint(0, n_arms)] = 12.0


iterations = 100
expected_rewards = [0.0] * n_arms # this should be the same size / shape as arm_rewards

# start with epsilon-greedy, then try some others
epsilon=0.2
exploration_mechanisms = {
    "Epsilon Greedy": lambda: explore_epsilon_greedy(epsilon),
    # "UCB": None,
    # "Thomson Sampling": None
} # all need to have the same lambda signature

reward_gained = 0
reward_possible = 0

explorations = 0
r_variation_potential = 0.1*rmax # maximum amount that a reward can vary by at each iteration

print(f"\nReward Range: ({rmin}, {rmax}); Max Drift Per Iteration: {r_variation_potential}")

# TODO: modify to add implementations for different exploration mechanisms, add loop around this for each
# could define a function for each exploration method that takes in the surrounding context (where necessary) and returns 0/1 whether or not to explore.
# store list of these functions, iterate these and print the corresponding results for each.

# TODO: sweep across different reward drift conditions. should start with stationary, and increase the maximum reward variation per iteration each time
# this would likely be the final / intended result for this mini-project
for explore_name, explore_f in exploration_mechanisms.items():
    for i in range(iterations):
        # # print(f"Arm Rewards (i={i}; n={n_arms}): {arm_rewards}")

        # use decided exploration mechanism to choose arm to pull
        explore = explore_f()
        
        if explore:
            explorations += 1
            arm = np.random.randint(0, n_arms) # choose which arm to pull at random
        else:
            arm = np.argmax(expected_rewards)

        # pull arm and get reward
        reward = arm_rewards[arm]
        # update expected reward with most recent (fine for this MAB since stationary and no future reward states)
        expected_rewards[arm] = reward
        
        # update total reward
        reward_gained += reward
        reward_possible += max(arm_rewards)

        # shift actual rewards at random
        for i in range(len(arm_rewards)):
            arm_rewards[i] = max(min(arm_rewards[i] + np.random.uniform(-r_variation_potential, r_variation_potential), rmax), rmin) # vary, constrained by rmax and rmin

    print(
        f"\n***{explore_name} Exploration***"
        f"\nReward: {reward_gained} of {reward_possible} possible ({reward_gained/reward_possible * 100}%)"
        f"\nRegret: {reward_possible - reward_gained}"
        f"\nExplorations: {explorations} of {iterations} iterations ({explorations/iterations * 100}%)"
    )
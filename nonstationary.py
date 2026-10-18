import numpy as np

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
exploration_mechanisms = []

# start with epsilon-greedy, then try some others
epsilon=0.2

reward_gained = 0
reward_possible = 0

explorations = 0
r_variation_potential = 0.1*rmax # maximum amount that a reward can vary by at each iteration

print(f"Reward Range: ({rmin}, {rmax}); Max Drift Per Iteration: {r_variation_potential}")

for i in range(iterations):
    # # print(f"Arm Rewards (i={i}; n={n_arms}): {arm_rewards}")

    # use decided exploration mechanism to choose arm to pull
    explore = np.random.binomial(n=1, p=epsilon) # epsilon-greedy

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
    f"***Epsilon Greedy Exploration with epsilon={epsilon}***"
    f"\nReward: {reward_gained} of {reward_possible} possible ({reward_gained/reward_possible * 100}%)"
    f"\nRegret: {reward_possible - reward_gained}"
    f"\nExplorations: {explorations} of {iterations} iterations ({explorations/iterations * 100}%)"
)
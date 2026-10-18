import numpy as np

n_arms = 100
rmin = 0
rmax = 100

# standard reward setup
arm_rewards = np.random.randint(rmin, rmax+1, size=n_arms)

# sparse reward setup
arm_rewards = [0] * n_arms
arm_rewards[np.random.randint(0, n_arms)] = 65
arm_rewards[np.random.randint(0, n_arms)] = 4
arm_rewards[np.random.randint(0, n_arms)] = 12

print(f"Arm Rewards (n={n_arms}): {arm_rewards}")

iterations = 1000
expected_rewards = [0] * n_arms # this should be the same size / shape as arm_rewards
exploration_mechanisms = []

# start with epsilon-greedy, then try some others
epsilon=0.2

reward_gained = 0
reward_possible = max(arm_rewards) * iterations

explorations = 0


for i in range(iterations):
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

print(
    f"***Epsilon Greedy Exploration with epsilon={epsilon}***"
    f"\nReward: {reward_gained} of {reward_possible} possible ({reward_gained/reward_possible * 100}%)"
    f"\nRegret: {reward_possible - reward_gained}"
    f"\nExplorations: {explorations} of {iterations} iterations ({explorations/iterations * 100}%)"
)
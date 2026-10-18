import numpy as np

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

exploration_mechanisms = {
    "Epsilon Greedy": lambda: explore_epsilon_greedy(epsilon),
    # "UCB": None,
    # "Thomson Sampling": None
} # all need to have the same lambda signature
drift_settings = [0*rmax, 0.01*rmax, 0.05*rmax, 0.1*rmax, 0.2*rmax, 0.4*rmax, 0.6*rmax, 0.8*rmax, 1.0*rmax] # maximum amount that a reward can vary by at each iteration

for r_variation_potential in sorted(drift_settings):
    print(
        f"\n----------------------"
        f"\nReward Range: ({rmin}, {rmax}); Max Drift Per Iteration: {r_variation_potential}"
    )
    for explore_name, explore_f in exploration_mechanisms.items():
        # initial settings setup
        reward_gained = 0
        reward_possible = 0
        explorations = 0
        arm_rewards = initial_arm_rewards
        expected_rewards = [0.0] * n_arms # this should be the same size / shape as arm_rewards
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
            # update expected reward with most recent (fine for this MAB since stationary and no future reward states)
            expected_rewards[arm] = reward
            
            # update total reward
            reward_gained += reward
            reward_possible += max(arm_rewards)

            # shift actual rewards at random
            for i in range(len(arm_rewards)):
                arm_rewards[i] = max(min(arm_rewards[i] + np.random.uniform(-r_variation_potential, r_variation_potential), rmax), rmin) # vary, constrained by rmax and rmin

        # TODO: add summary data to some structure / dictionary object for later results figure
        print(
            f"\n***{explore_name} Exploration***"
            f"\nReward: {reward_gained} of {reward_possible} possible ({reward_gained/reward_possible * 100}%)"
            f"\nRegret: {reward_possible - reward_gained}"
            f"\nExplorations: {explorations} of {iterations} iterations ({explorations/iterations * 100}%)"
        )
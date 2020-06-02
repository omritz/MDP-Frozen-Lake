# -*- coding: utf-8 -*-

from World import World
import numpy as np
import matplotlib.pyplot as plt


def action_values(mdp, state, V, gamma, r):
    A = np.zeros(mdp.nActions)
    for a in range(mdp.nActions):
        for i in range(4):
            prob, next_state, reward, done = mdp.step(state, a, i, r)
            # print('Action: %s, i: %s, State: %s, Probability: %s, Next state: %s, Reward: %s, Done: %s'
            #       % (a, i, state, prob, next_state, reward, done))
            A[a] += prob * (reward + gamma * V[next_state-1])
    return A


def value_iteration(mdp, theta, gamma, r, max_iterations=10000):
    V = np.zeros(mdp.nStates)
    for i in range(max_iterations):
        delta = 0
        # Update each state...
        for s in range(mdp.nStates):
            # Do a one-step lookahead to find the best action
            A = action_values(mdp, s, V, gamma, r)
            best_action_value = np.max(A)
            # print('A: %s' % A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function.
            V[s] = best_action_value
        # Stopping condition
        if delta < theta:
            break
    # print('V: %s' % V)
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([mdp.nStates])
    for s in range(mdp.nStates):
        # One step lookahead to find the best action for this state
        A = action_values(mdp, s, V, gamma, r)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s] = best_action+1

    # print(policy)
    return policy, V


def policy_eval(policy, mdp, theta, gamma, r, max_iterations=10000):

    # Start with a random (all 0) value function
    V = np.zeros(mdp.nStates)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(mdp.nStates):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for i in range(4):
                    prob, next_state, reward, done = mdp.step(s, a, i, r)
                    # Calculate the expected value
                    v += action_prob * prob * (reward + gamma * V[next_state-1])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(mdp, policy, V, gamma, r):

    policy_stable = True
    for s in range(mdp.nStates):
        # The best action we would take under the current policy
        chosen_a = np.argmax(policy[s])
        # Find the best action by one-step lookahead
        actionValues = action_values(mdp, s, V, gamma, r)
        # print('action_values: %s' % action_values)
        best_a = np.argmax(actionValues)
        # Greedily update the policy
        if chosen_a != best_a:
            policy_stable = False
        policy[s] = np.eye(mdp.nActions)[best_a]
    return policy_stable


def policy_iteration(mdp, theta, gamma, r):

    # Start with a random policy
    policy = np.ones([mdp.nStates, mdp.nActions]) / mdp.nActions
    while True:
        # Evaluate the current policy
        V = policy_eval(policy, mdp, theta, gamma, r)
        policy_stable = policy_improvement(mdp, policy, V, gamma, r)
        mdp.plot_value(V)
        mdp.plot_policy(np.array(np.argmax(policy, axis=1)+1))
        # If the policy is stable we've found an optimal policy. Return it
        # print(policy)
        if policy_stable:
            return policy, V


def plot_transition_matrix(env, action):
    transition_matrix = np.zeros([16, 16])
    for s in range(env.nStates):
        for a in range(env.nActions):
            prob, next_state, reward, done = env.step(s, action, a, 1)
            transition_matrix[s][next_state - 1] = prob
    # print(transition_matrix)
    data = transition_matrix
    row_labels = [i + 1 for i in range(16)]
    col_labels = [i + 1 for i in range(16)]
    matrix = plt.table(cellText=data, rowLabels=row_labels, colLabels=col_labels, loc="center right")
    matrix.set_fontsize(7)
    plt.axis("off")
    actions = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
    plt.title('p(s\'| s, %s)' % actions[action])
    plt.show()


if __name__ == "__main__":
    exercises = [(1, -0.04), (0.9, -0.04), (1, -0.02)]  # (gamma, reward)
    env = World()
    for i in range(env.nActions):
        plot_transition_matrix(env, i)
    theta = 10**-4
    for gamma, reward in exercises:
        p, v = value_iteration(env, theta, gamma, reward)
        env.plot_value(v)
        env.plot_policy(p)
    p, v = policy_iteration(env, theta, 0.9, -0.04)




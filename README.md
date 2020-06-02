# MDP-Frozen-Lake
Solving Frozen Lake MDP with value iteration and policy iteration algorithms.


Given is a penguin on a frozen lake, which is described by a 4x4 grid world with holes and a goal state (fish),
both defining terminal states. For transitions to terminal states the penguin gets a reward of +1 for the
goal state and a reward of −1 for the holes, whereas for all other transitions the penguin gets a reward of
r = −0.04. The penguin can take four possible actions = {N, E, S, W}, but the surface is slippery and only
with probability 0.8 the penguin will go into the intended direction and with probability 0.1 to the left and
0.1 to the right of it. It is assumed that the boundaries are reflective, i.e., if the penguin hits the edge of
the lake it remains in the same grid location. Find the optimal policy for the penguin to get to the fish by
solving the corresponding MDP.

# Robot Navigation in a 1D Hallway using Markov Decision Processes (MDP)

**Subject:** Reinforcement Learning & Optimal Control
**Architecture:** MDP Solver (Value Iteration)
**Simulation:** Python Tkinter GUI

## 📌 Project Overview

The objective of this project is to model and solve a sequential decision-making problem: finding the optimal movement strategy (the **policy**) for a robot operating in a simple 1D hallway environment. This is modeled as a **Markov Decision Process (MDP)**, where the robot seeks to maximize its accumulated discounted reward by reaching a charging station (the Goal State) as efficiently as possible.

The optimal strategy is found using the **Value Iteration algorithm**, a fundamental method in reinforcement learning and dynamic programming. The final optimal policy ($\pi^*$) is then used to control the robot in a **Tkinter-based GUI simulation** for visual demonstration.

## 💡 Key Concepts

This project is built upon the following theoretical concepts:

### Markov Decision Process (MDP)
An MDP is a formal framework for sequential decision-making in a stochastic (random) environment. It is defined by:
* **States ($S$):** The possible positions of the robot in the hallway.
* **Actions ($A$):** The deterministic moves the robot can attempt (LEFT or RIGHT).
* **Transition Probability ($T(s'|s, a)$):** The probability of moving to a new state $s'$ given the current state $s$ and the action taken $a$.
* **Reward Function ($R(s, a, s')$):** The immediate reward received for a transition.
* **Discount Factor ($\gamma$):** A parameter determining the importance of future rewards compared to immediate rewards.



### Value Iteration
Value Iteration is an iterative algorithm used to find the **optimal value function ($V^*$)** for an MDP. It works by repeatedly applying the **Bellman Optimality Equation** until the state values converge below a specified threshold ($\Theta$).

The algorithm's core is the Bellman Optimality Update, which computes the maximum expected return from a state $s$:

$$V_{k+1}(s) = \max_a \sum_{s'} T(s'|s, a) [R(s, a, s') + \gamma V_k(s')]$$

The final result, $V^*$, represents the maximum expected discounted future reward the robot can achieve starting from state $s$.

## ⚙️ System Configuration

The modeled environment is a simple 1D hallway with four positions (State 0 to State 3):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **States ($S$)** | {0, 1, 2, 3} | Four positions in the hallway. |
| **Terminal State** | State 3 | The charging station (Goal). |
| **Discount Factor ($\gamma$)** | 0.9 | Future rewards are discounted by 10% per step. |
| **Move Success ($P_{MOVE}$)** | 0.8 | The chosen action succeeds with 80% probability. |
| **Reward upon Goal Arrival** | +9.0 | Calculated as $R_{goal} (10.0) + R_{step} (-1.0)$. |
| **Step Cost ($R_{step}$)** | -1.0 | Cost incurred for every move attempt to a non-terminal state. |

## 🛠️ Methodology

### 1. Transition Model Implementation
The `transition_probability(s_prime, s, a)` function implements the stochastic environment:
* **Successful Move:** The robot transitions to the intended state with probability $P_{MOVE}=0.8$.
* **Failed Move:** The robot stays in its current state with probability $1 - P_{MOVE}=0.2$.
* **Boundary Handling:** If an action would cause the robot to move off the map (e.g., LEFT from State 0), the robot is forced to stay at the boundary state with 100% probability (a "blocked move").

### 2. Optimal Policy Derivation
The project flow involves:
1.  **Value Iteration:** Calculating the optimal value function $V^*$.
2.  **Policy Extraction:** Using the converged $V^*$ to calculate the Q-value for every action in every state, and selecting the action that maximizes the expected return.
3.  **GUI Simulation:** The final policy is loaded into the `RobotSimulationGUI` (Tkinter) to allow a user to step through an episode, visually demonstrating the optimal actions and the stochastic movement of the robot.

## 📊 Results

The Value Iteration algorithm successfully converged to the following optimal value function and policy, using the parameters $\gamma=0.9$ and $P_{MOVE}=0.8$:

| State ($s$) | Optimal Action $\pi^*(s)$ | Value $V^*(s)$ |
| :--- | :--- | :--- |
| **0** | `RIGHT` | 11.0606 |
| **1** | `RIGHT` | 13.9857 |
| **2** | `RIGHT` | 17.3171 |
| **3** | `STAY` (Terminal) | 10.0000 |

*The resulting $V^*(s)$ values indicate the high expected long-term reward gained by starting in state $s$ and acting optimally thereafter, driving the robot toward the goal.*

## 🛠️ Dependencies

To run this project, you need the following Python libraries:

```bash
# Core libraries
import numpy as np
import random
import sys

# GUI library
import tkinter as tk

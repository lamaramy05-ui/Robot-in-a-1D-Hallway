import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import sys

# --- 1. MDP Setup and Parameters ---

# States: S = {0, 1, 2, 3}
STATES = [0, 1, 2, 3]
# Actions: A = {LEFT, RIGHT}
ACTIONS = ['LEFT', 'RIGHT']
# Terminal State
TERMINAL_STATE = 3

# Assumed Model Parameters:
GAMMA = 0.9     # Discount Factor
THETA = 1e-6    # Convergence threshold
P_MOVE = 0.8    # Probability of moving in the chosen direction
GOAL_REWARD = 10.0
STEP_COST = -1.0

# --- 2. Core MDP Functions ---

def get_reward_for_step(current_state, next_state):
    """
    Calculates the reward for the transition from current_state to next_state.
    This accounts for the -1 move cost and the +10 goal reward upon arrival.
    """
    reward = STEP_COST # Every move costs -1
    
    if next_state == TERMINAL_STATE:
        reward += GOAL_REWARD # Add +10 upon reaching the terminal state
        
    return reward

def get_intended_next_state(current_state, action):
    """
    Determines the position the robot *intends* to move to,
    handling the boundary conditions.
    """
    if current_state == TERMINAL_STATE:
        return current_state
        
    if action == 'RIGHT':
        return min(current_state + 1, TERMINAL_STATE)
    elif action == 'LEFT':
        return max(current_state - 1, 0)
    else:
        return current_state 

def transition_probability(s_prime, s, a):
    """
    T(s' | s, a): Probability of transitioning to state s' from state s 
    when taking action a.
    """
    if s == TERMINAL_STATE:
        return 1.0 if s_prime == s else 0.0

    s_intended = get_intended_next_state(s, a)
    s_stay = s
    
    # 1. Successful move (P_MOVE)
    if s_prime == s_intended:
        if s_intended == s_stay:
            return 1.0 # Blocked move: 100% stays at s
        else:
            return P_MOVE
            
    # 2. Failed move (1 - P_MOVE)
    elif s_prime == s_stay:
        if s_intended != s_stay:
            return 1.0 - P_MOVE
        else:
            return 0.0
            
    return 0.0


# --- 3. Value Iteration Algorithm (Fixed Reward Structure) ---

def value_iteration():
    """
    Iteratively solves the Bellman Optimality Equation until convergence.
    """
    V = {s: 0.0 for s in STATES}
    V[TERMINAL_STATE] = GOAL_REWARD 
    
    iteration = 0
    
    while True:
        delta = 0 
        V_old = V.copy()
        
        for s in STATES:
            if s == TERMINAL_STATE:
                V[s] = GOAL_REWARD
                continue
                
            q_values = {}
            for a in ACTIONS:
                expected_return = 0
                for s_prime in STATES:
                    prob = transition_probability(s_prime, s, a)
                    reward_sa_sprime = get_reward_for_step(s, s_prime)
                    expected_return += prob * (reward_sa_sprime + GAMMA * V_old[s_prime])
                
                q_values[a] = expected_return
            
            V[s] = max(q_values.values())
            delta = max(delta, abs(V[s] - V_old[s]))
            
        iteration += 1
        
        if delta < THETA:
            break
            
    print(f"Value Iteration converged after {iteration} iterations.")
    return V

def extract_optimal_policy(V):
    """
    Derives the optimal policy pi* from the converged optimal value function V*.
    """
    policy = {}
    
    for s in STATES:
        if s == TERMINAL_STATE:
            policy[s] = 'STAY (Terminal)'
            continue
            
        q_values = {}
        for a in ACTIONS:
            expected_return = 0
            for s_prime in STATES:
                prob = transition_probability(s_prime, s, a)
                reward_sa_sprime = get_reward_for_step(s, s_prime)
                expected_return += prob * (reward_sa_sprime + GAMMA * V[s_prime])
            
            q_values[a] = expected_return
            
        best_action = max(q_values, key=q_values.get)
        policy[s] = best_action
        
    return policy

# --- 4. GUI Simulation Class ---

class RobotSimulationGUI:
    def __init__(self, master, optimal_V, optimal_policy):
        self.master = master
        master.title("Robot in a 1D Hallway (MDP Simulation)")
        
        # MDP Results
        self.optimal_V = optimal_V
        self.optimal_policy = optimal_policy
        
        # Simulation State
        self.current_state = random.choice([0, 1, 2]) # Start at a random non-terminal state
        self.episode_steps = 0
        self.total_reward = 0
        
        self.setup_gui()
        self.update_gui()

    def setup_gui(self):
        # --- Hallway Visualization Frame ---
        hallway_frame = tk.Frame(self.master)
        hallway_frame.pack(pady=20)
        
        self.state_labels = []
        for i in STATES:
            # Base Text: Pos and Goal info only
            base_text = f"Pos {i}"
            if i == TERMINAL_STATE:
                base_text += "\n(GOAL: +10)"
            
            label = tk.Label(hallway_frame, text=base_text, width=15, height=4, 
                             borderwidth=3, relief="raised", font=('Arial', 10, 'bold'))
            label.grid(row=0, column=i, padx=5)
            self.state_labels.append(label)

        # --- Status and Controls ---
        
        # *** START REMOVED SECTION ***
        # The Optimal Policy Display Header and Table Frame are removed here.
        # This removes the V table from the GUI as requested.
        
        # Status Label
        self.status_label = tk.Label(self.master, text="", pady=10, font=('Arial', 10, 'bold'))
        self.status_label.pack()
        
        # Probability/Action Detail Label (previously removed in prior steps)
        # We must keep the placeholder in setup_gui if it is referenced elsewhere,
        # but since we removed its updates in the last step, we can remove its packing now.
        # self.detail_label = tk.Label(self.master, text="", pady=5, font=('Arial', 9), fg='navy', justify=tk.LEFT)
        # self.detail_label.pack() 

        # Control Buttons Frame
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="Take Optimal Step", command=self.take_step, width=20).pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="Reset Simulation", command=self.reset_simulation, width=20).pack(side=tk.LEFT, padx=10)

    # *** display_policy_table function is removed as it is no longer used. ***
    # It has been removed to reduce the code length and remove the V table display.
    # The extraction logic remains in the core MDP functions.

    def take_step(self):
        if self.current_state == TERMINAL_STATE:
            messagebox.showinfo("Goal Reached", "The robot is at the charging station (Pos 3). Please reset to start a new episode.")
            return

        old_state = self.current_state
        action = self.optimal_policy[old_state]
        
        # 1. Determine the transition based on probability (Monte Carlo)
        s_intended = get_intended_next_state(old_state, action)
        
        # Monte Carlo sampling: 0.8 to intended, 0.2 to stay
        next_state = old_state 
        
        # Check for blocked move (100% stay)
        if transition_probability(old_state, old_state, action) == 1.0 and old_state == s_intended:
             next_state = old_state
        # Check for successful move (P_MOVE)
        elif np.random.rand() < P_MOVE:
            next_state = s_intended
        # Otherwise, failed move (stays put)
        else:
            next_state = old_state

        # 2. Update state and reward
        step_reward = get_reward_for_step(old_state, next_state) 
        self.total_reward += step_reward
        self.episode_steps += 1
        
        self.current_state = next_state
        
        # 3. Update GUI
        self.update_gui()
        
        if self.current_state == TERMINAL_STATE:
            messagebox.showinfo("Goal Reached", 
                                f"Episode finished! Steps: {self.episode_steps}, Total Reward: {self.total_reward:.2f}.")

    def reset_simulation(self):
        start_state = random.choice([0, 1, 2])
        self.current_state = start_state
        self.episode_steps = 0
        self.total_reward = 0
        self.update_gui()

    def update_gui(self, *args):
        
        # --- A. Update Hallway Labels (Highlighting) ---
        for i, label in enumerate(self.state_labels):
            is_robot_here = (i == self.current_state)
            
            # Base Text: Pos and Goal info only
            base_text = f"Pos {i}"
            if i == TERMINAL_STATE:
                base_text += "\n(GOAL: +10)"
            
            # Colors/Relief
            bg_color = 'blue' if is_robot_here else ('green' if i == TERMINAL_STATE else 'SystemButtonFace')
            fg_color = 'white' if is_robot_here or i == TERMINAL_STATE else 'black'
            relief_type = "groove" if is_robot_here else ("sunken" if i == TERMINAL_STATE else "raised")
            
            if is_robot_here:
                label.config(text=f"{base_text}\n(ROBOT HERE)", bg=bg_color, fg=fg_color, relief=relief_type)
            else:
                 label.config(text=base_text, bg=bg_color, fg=fg_color, relief=relief_type)

        # --- B. Update Status Label ---
        status_text = f"Current Pos: **{self.current_state}** | Steps: **{self.episode_steps}** | Total Reward: **{self.total_reward:.2f}**"
        self.status_label.config(text=status_text)


# --- 5. Main Execution ---

if __name__ == "__main__":
    
    # 1. Find the optimal Value Function V*
    optimal_V = value_iteration()
    
    # 2. Extract the Optimal Policy pi*
    optimal_policy = extract_optimal_policy(optimal_V)
    
    # 3. Print the results (still print in console as it's useful for debugging)
    print("\n### Optimal Policy and Value Function ###")
    print(f"Discount Factor (γ): {GAMMA}, Move Success Probability: {P_MOVE}")
    print("\n| State (s) | Optimal Action $\pi^*(s)$ | Value $V^*(s)$ |")
    print("|-----------|-------------------------|----------------|")
    for s, a in optimal_policy.items():
        print(f"| {s}         | {a}                     | {optimal_V[s]:.4f}         |")
    
    # 4. Start the GUI
    print("\nStarting GUI Simulation...")
    root = tk.Tk()
    sim = RobotSimulationGUI(root, optimal_V, optimal_policy)
    root.mainloop()
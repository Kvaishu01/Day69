# Day 69: Simulated Quenching Optimizer 🔥
# Author: Vaishnavi
# Goal: Demonstrate faster convergence optimization using Simulated Quenching

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Objective Function (Sphere)
# ---------------------------
def objective_function(x):
    return np.sum(x ** 2)

# ---------------------------
# Simulated Quenching Algorithm
# ---------------------------
def simulated_quenching(objective_func, dim=2, bounds=(-5, 5), max_iter=500, T_start=100, T_end=1e-3, alpha=0.85):
    # Initialize random solution
    current_x = np.random.uniform(bounds[0], bounds[1], dim)
    current_f = objective_func(current_x)
    
    best_x, best_f = current_x.copy(), current_f
    T = T_start
    history = []

    for i in range(max_iter):
        # Generate new candidate
        new_x = current_x + np.random.uniform(-1, 1, dim)
        new_x = np.clip(new_x, bounds[0], bounds[1])
        new_f = objective_func(new_x)

        # Acceptance criterion
        delta = new_f - current_f
        if delta < 0 or np.exp(-delta / T) > np.random.rand():
            current_x, current_f = new_x, new_f

        # Update best
        if new_f < best_f:
            best_x, best_f = new_x, new_f

        # Store history
        history.append(best_f)

        # Faster cooling (quenching)
        T *= alpha

        # Stop early
        if T < T_end:
            break

    return best_x, best_f, history

# ---------------------------
# Streamlit App
# ---------------------------
st.title("🔥 Simulated Quenching Optimization (Day 69)")
st.markdown("### Fast metaheuristic inspired by annealing — with accelerated cooling")

dim = st.slider("Select number of dimensions", 1, 10, 2)
max_iter = st.slider("Max Iterations", 100, 2000, 500)
alpha = st.slider("Cooling Rate (alpha)", 0.7, 0.99, 0.85)

if st.button("Run Optimization 🚀"):
    best_x, best_f, history = simulated_quenching(objective_function, dim=dim, max_iter=max_iter, alpha=alpha)
    
    st.success(f"✅ Best solution found: {best_x}")
    st.success(f"🎯 Best fitness: {best_f:.6f}")

    # Plot convergence
    fig, ax = plt.subplots()
    ax.plot(history, label="Best Fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness Value")
    ax.set_title("Convergence Curve - Simulated Quenching")
    ax.legend()
    st.pyplot(fig)

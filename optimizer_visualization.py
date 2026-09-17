import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Streamlit Optimization Method Explorer", layout="wide")

st.title("Streamlit Optimization Method Explorer")
st.header("Optimizer Path Comparison")
st.markdown("""
* Non-linear data fitting problem: $y = 2.0 e^{1.5x}$
* Synthetic noisy data generated for demonstration.
""")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
start_x = st.sidebar.slider("Starting point X", min_value=-2.0, max_value=4.0, value=2.0, step=0.1)
start_y = st.sidebar.slider("Y coordinate: Y", min_value=-2.0, max_value=4.0, value=1.0, step=0.1)
max_iter = st.sidebar.slider("Iteration step (max iterations)", min_value=0, max_value=20, value=20, step=1)

st.sidebar.markdown("Which optimization paths are:")
show_gd = st.sidebar.checkbox("Gradient Descent", value=True)
show_nw = st.sidebar.checkbox("Newton's Method", value=True)
show_gn = st.sidebar.checkbox("Gauss-Newton", value=True)
show_lm = st.sidebar.checkbox("dynamic Levenberg-Marquardt", value=False)

initial_lambda = st.sidebar.slider("Initial Levenberg-Marquardt damping", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
learning_rate = st.sidebar.slider("Gradient Descent learning rate", min_value=0.001, max_value=0.2, value=0.05, step=0.001)

# --- Math & Data ---
np.random.seed(42)
x_data = np.linspace(0, 1, 10)
true_a, true_b = 2.0, 1.5
y_data = true_a * np.exp(true_b * x_data) + np.random.normal(0, 0.1, len(x_data))

def model(p, x):
    return p[0] * np.exp(p[1] * x)

def residuals(p):
    return model(p, x_data) - y_data

def jacobian(p):
    J = np.zeros((len(x_data), 2))
    J[:, 0] = np.exp(p[1] * x_data)                  
    J[:, 1] = p[0] * x_data * np.exp(p[1] * x_data)  
    return J

def exact_hessian(p):
    res = residuals(p)
    J = jacobian(p)
    H_GN = J.T @ J
    H_sec = np.zeros((2, 2))
    for i in range(len(x_data)):
        x_i = x_data[i]
        H_i = np.zeros((2, 2))
        H_i[0, 1] = x_i * np.exp(p[1] * x_i)
        H_i[1, 0] = H_i[0, 1]
        H_i[1, 1] = p[0] * (x_i**2) * np.exp(p[1] * x_i)
        H_sec += res[i] * H_i
    return H_GN + H_sec

def get_cost(p):
    return 0.5 * np.sum(residuals(p)**2)

# --- Optimizers ---
p_init = np.array([start_x, start_y])

# 1. Gradient Descent
steps_GD = [p_init.copy()]
p = p_init.copy()
for _ in range(max_iter):
    J = jacobian(p)
    grad = J.T @ residuals(p)
    p = p - (learning_rate * grad)
    steps_GD.append(p.copy())

# 2. Gauss-Newton
steps_GN = [p_init.copy()]
p = p_init.copy()
for _ in range(max_iter):
    J = jacobian(p)
    try:
        step = np.linalg.solve(J.T @ J, J.T @ residuals(p))
        p = p - step
    except np.linalg.LinAlgError:
        pass # Handle singular matrix if it diverges wildly
    steps_GN.append(p.copy())

# 3. Newton's Method
steps_Newton = [p_init.copy()]
p = p_init.copy()
for _ in range(max_iter):
    J = jacobian(p)
    try:
        step = np.linalg.solve(exact_hessian(p), J.T @ residuals(p))
        p = p - step
    except np.linalg.LinAlgError:
        pass
    steps_Newton.append(p.copy())

# 4. Dynamic Levenberg-Marquardt
lambda_param = initial_lambda
steps_LM = [p_init.copy()]
p = p_init.copy()

for _ in range(max_iter):
    J = jacobian(p)
    res = residuals(p)
    current_cost = get_cost(p)
    
    JTJ = J.T @ J
    H_LM = JTJ + lambda_param * np.diag(np.diag(JTJ))
    grad = J.T @ res
    
    try:
        step = np.linalg.solve(H_LM, grad)
        p_try = p - step
        
        if get_cost(p_try) < current_cost:
            p = p_try
            lambda_param /= 10.0
        else:
            lambda_param *= 10.0
    except np.linalg.LinAlgError:
        lambda_param *= 10.0
        
    steps_LM.append(p.copy())

steps_GD = np.array(steps_GD)
steps_GN = np.array(steps_GN)
steps_Newton = np.array(steps_Newton)
steps_LM = np.array(steps_LM)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

if show_gd:
    ax.plot(steps_GD[:, 0], steps_GD[:, 1], 'm-', label="Gradient Descent", linewidth=1.5, alpha=0.8)
if show_nw:
    ax.plot(steps_Newton[:, 0], steps_Newton[:, 1], 'g--o', label="Newton's Method", linewidth=1.5)
if show_gn:
    ax.plot(steps_GN[:, 0], steps_GN[:, 1], 'b-^', label="Gauss-Newton", linewidth=1.5)
if show_lm:
    ax.plot(steps_LM[:, 0], steps_LM[:, 1], color='darkorange', marker='v', linestyle='-', label="dynamic Levenberg-Marquardt", linewidth=1.5)

# True minimum indicator
ax.plot(true_a, true_b, 'r*', markersize=10, label="True Minimum")

# Styling
ax.grid(True, linestyle='-', color='0.8')
ax.set_xlim(-2.5, 4.5)
ax.set_ylim(-2.5, 4.5)
ax.legend(loc='upper left', frameon=True)

st.pyplot(fig)

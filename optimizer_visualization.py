import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Streamlit Optimization Method Explorer", layout="wide")

st.title("Streamlit Optimization Method Explorer")
st.header("Optimizer Path Comparison")
st.markdown("""
* Non-linear data fitting problem: $y = 2.0 e^{1.5x}$
* Synthetic noisy data generated for demonstration.
* **Scroll to zoom, drag to pan.**
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
        pass 
    steps_GN.append(p.copy())

# 3. Newton's Method
steps_Newton = [p_init.copy()]
p = p_init.copy()
for _ in range(max_iter):
    try:
        step = np.linalg.solve(exact_hessian(p), jacobian(p).T @ residuals(p))
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

# --- Calculate Contour Grid ---
# Removed logarithmic scaling, now using raw linear cost values
a_range = np.linspace(-2.5, 4.5, 60)
b_range = np.linspace(-2.5, 4.5, 60)
A, B = np.meshgrid(a_range, b_range)
Z = np.zeros_like(A)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Z[i, j] = get_cost([A[i, j], B[i, j]])

# --- Plotting with Plotly ---
fig = go.Figure()

# Add Contour surface
fig.add_trace(go.Contour(
    x=a_range, y=b_range, z=Z,
    colorscale='Viridis',
    opacity=0.4,
    showscale=False,
    hoverinfo='skip',
    # Optional: You can uncomment the line below if you want Plotly to draw more contour lines dynamically
    # ncontours=30 
))

# Add Optimizer Paths
if show_gd:
    fig.add_trace(go.Scatter(
        x=steps_GD[:, 0], y=steps_GD[:, 1],
        mode='lines+markers', name="Gradient Descent",
        line=dict(color='magenta', width=2),
        marker=dict(symbol='square', size=6)
    ))

if show_nw:
    fig.add_trace(go.Scatter(
        x=steps_Newton[:, 0], y=steps_Newton[:, 1],
        mode='lines+markers', name="Newton's Method",
        line=dict(color='green', width=2, dash='dash'),
        marker=dict(symbol='circle', size=6)
    ))

if show_gn:
    fig.add_trace(go.Scatter(
        x=steps_GN[:, 0], y=steps_GN[:, 1],
        mode='lines+markers', name="Gauss-Newton",
        line=dict(color='blue', width=2),
        marker=dict(symbol='triangle-up', size=7)
    ))

if show_lm:
    fig.add_trace(go.Scatter(
        x=steps_LM[:, 0], y=steps_LM[:, 1],
        mode='lines+markers', name="dynamic Levenberg-Marquardt",
        line=dict(color='darkorange', width=2),
        marker=dict(symbol='triangle-down', size=7)
    ))

# Add True Minimum
fig.add_trace(go.Scatter(
    x=[true_a], y=[true_b],
    mode='markers', name="True Minimum",
    marker=dict(color='red', symbol='star', size=12)
))

# Style and adjust height
fig.update_layout(
    height=550,  
    xaxis_title="Parameter a",
    yaxis_title="Parameter b",
    xaxis=dict(range=[-2.5, 4.5], zeroline=False),
    yaxis=dict(range=[-2.5, 4.5], zeroline=False),
    legend=dict(
        yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)"
    ),
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

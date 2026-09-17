import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="LM Algorithm Optimizer", layout="centered")

st.title("LM Algorithm Optimizer")

# We create layout containers to force the visual order (Plot -> Metrics -> Controls)
# while allowing the script to calculate data top-to-bottom.
plot_container = st.container()
metrics_container = st.container()
st.write("") # Spacer

# --- CONTROLS (Rendered at the bottom) ---
start_x = st.slider("Start X", min_value=-5.0, max_value=5.0, value=-2.0, step=0.1)
start_y = st.slider("Start Y", min_value=-5.0, max_value=5.0, value=-2.4, step=0.1)
current_iter = st.slider("Iteration Step", min_value=0, max_value=20, value=20, step=1)
initial_lambda = st.slider("Initial LM Lambda", min_value=0.001, max_value=10.0, value=1.0, step=0.1)
learning_rate = st.slider("GD Learning Rate", min_value=0.001, max_value=0.2, value=0.05, step=0.001)

st.write("---")

show_nw = st.toggle("Show Newton", value=True)
show_gn = st.toggle("Show Gauss-Newton", value=True)
show_lm = st.toggle("Show Levenberg-Marquardt", value=True)
show_gd = st.toggle("Show Gradient Descent", value=False)

# --- MATH & DATA ---
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

# --- OPTIMIZERS ---
p_init = np.array([start_x, start_y])
max_iter = current_iter  # Only calculate up to the selected slider step

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
lambda_history = [lambda_param]
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
    lambda_history.append(lambda_param)

steps_GD = np.array(steps_GD)
steps_GN = np.array(steps_GN)
steps_Newton = np.array(steps_Newton)
steps_LM = np.array(steps_LM)

# --- CALCULATE CONTOUR ---
# Expanded view bounds to match the -5 to 5 scale in the reference image
a_range = np.linspace(-5, 5, 80)
b_range = np.linspace(-5, 5, 80)
A, B = np.meshgrid(a_range, b_range)
Z = np.zeros_like(A)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Z[i, j] = get_cost([A[i, j], B[i, j]])

# --- BUILD PLOTLY CHART ---
fig = go.Figure()

# Add wireframe-style Contour surface
fig.add_trace(go.Contour(
    x=a_range, y=b_range, z=Z,
    contours_coloring='lines', # Replaces solid fill with clean lines
    line_width=1,
    colorscale='Greys',
    opacity=0.3,
    showscale=False,
    ncontours=45,
    hoverinfo='skip'
))

# Add Optimizer Paths
if show_gd:
    fig.add_trace(go.Scatter(
        x=steps_GD[:, 0], y=steps_GD[:, 1],
        mode='lines+markers', name="Gradient Descent",
        line=dict(color='#E324B3', width=2, dash='dot'),
        marker=dict(size=6)
    ))

if show_nw:
    fig.add_trace(go.Scatter(
        x=steps_Newton[:, 0], y=steps_Newton[:, 1],
        mode='lines+markers', name="Newton",
        line=dict(color='#F2A93B', width=2),
        marker=dict(size=6)
    ))

if show_gn:
    fig.add_trace(go.Scatter(
        x=steps_GN[:, 0], y=steps_GN[:, 1],
        mode='lines+markers', name="Gauss-Newton",
        line=dict(color='#32A852', width=2, dash='dash'),
        marker=dict(size=6)
    ))

if show_lm:
    fig.add_trace(go.Scatter(
        x=steps_LM[:, 0], y=steps_LM[:, 1],
        mode='lines+markers', name="Levenberg-Marquardt",
        line=dict(color='#4B93FF', width=2),
        marker=dict(size=6)
    ))

# Add True Minimum
fig.add_trace(go.Scatter(
    x=[true_a], y=[true_b],
    mode='markers', name="True Minimum",
    showlegend=False,
    marker=dict(color='white', line=dict(color='#32A852', width=2), symbol='circle-dot', size=12)
))

# Styling matching the image
fig.update_layout(
    height=500,  
    xaxis_title="X Position →",
    yaxis_title="Y Position ↑",
    xaxis=dict(range=[-5, 5], zeroline=False, gridcolor='rgba(200,200,200,0.2)'),
    yaxis=dict(range=[-5, 5], zeroline=False, gridcolor='rgba(200,200,200,0.2)'),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.2,
        xanchor="left", x=0
    ),
    margin=dict(l=0, r=0, t=10, b=0)
)

# Render Plot in the top container
plot_container.plotly_chart(fig, use_container_width=True)

# Render Metrics in the middle container
col1, col2 = metrics_container.columns(2)
col1.metric("CURRENT LAMBDA", f"{lambda_history[-1]:.2e}")
col2.metric("COST FUNCTION", f"{get_cost(steps_LM[-1]):.2f}")

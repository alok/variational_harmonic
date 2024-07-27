#!/usr/bin/env python3
#%%
from math import floor
import torch
from jaxtyping import Float
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

"""given interval from 0 to 1 and a curve and a functional, minimize it

Imagine a chain of functions where for all f, f(0) = 0, f(1) = 1.

Each function is represented as a piecewise linear function, idealized as a series of infinitesimally small line segments (linelets).

Then the length of the chain is the sum of the lengths of all the linelets.

2nd order diffeq bc chain of funcs anchored at boundary conditions (start/end points)
"""

start = 0
end = 1

# a hyperfinite number, representing the length of the chain
H = 2**10
# create a random connected curve by brownian motion
# create random offsets dx 
f = torch.zeros(H)
for i in range(1,H):
    f[i] = f[i-1] + torch.randn(1).item()
# Normalize f to start at 0 and end at 1
f = (f - f[0]) / (f[-1] - f[0])

xs = torch.linspace(0, 1, H)

# Compute arclength using dot product and Euclidean distance
def compute_arclength(xs: Float[torch.Tensor, "H"], ys: Float[torch.Tensor, "H"]) -> Float[torch.Tensor, "1"]:
    """
    Compute the arclength of a curve defined by x and y coordinates using dot product and Euclidean distance.
    
    Args:
    xs (torch.Tensor): x-coordinates of the curve
    ys (torch.Tensor): y-coordinates of the curve
    
    Returns:
    float: The total arclength of the curve
    """
    # Combine x and y coordinates into a single tensor of shape (H, 2)
    points = torch.stack((xs, ys), dim=1)
    
    # Compute differences between consecutive points
    diffs = points[1:] - points[:-1]
    
    # Compute Euclidean distances using dot product
    # (a · a)^0.5 is equivalent to the Euclidean norm of a
    segment_lengths = torch.sqrt(torch.sum(diffs * diffs, dim=1))
    
    # Sum up all segment lengths to get total arclength
    total_arclength = torch.sum(segment_lengths)
    
    return total_arclength

# Optimize the curve to minimize arclength
def optimize_curve(xs, ys, num_iterations=30000, learning_rate=3e-4):
    # Create a tensor of y-coordinates that we'll optimize, excluding endpoints
    ys_opt = torch.nn.Parameter(ys[1:-1].clone())
    full_ys = torch.cat([ys[:1], ys_opt, ys[-1:]])
    # Use Adam optimizer
    optimizer = torch.optim.AdamW([ys_opt], lr=learning_rate)
    
    arclengths = []
    curves = []
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Reconstruct full curve with fixed endpoints
        full_ys = torch.cat([ys[:1], ys_opt, ys[-1:]])
        
        # Compute the current arclength
        arclength = compute_arclength(xs, full_ys)
        
        # We want to minimize the arclength
        loss = arclength
        
        # Compute gradients and update parameters
        loss.backward()
        optimizer.step()
        
        # Store arclength and curve for animation
        if i % 100 == 0:
            arclengths.append(arclength.item())
            curves.append(full_ys.detach().clone())
        
        # Print progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"Iteration {i+1}, Arclength: {arclength.item():.6f}")
    
    return full_ys, arclengths, curves

# Optimize the curve
optimized_ys, arclengths, curves = optimize_curve(xs, f)

# Create a visualization using ipywidgets and plotly
def create_plot(frame) -> go.Figure:
    fig = go.Figure()
    
    # Original curve
    fig.add_trace(go.Scatter(x=xs.numpy(), y=f.numpy(), mode='lines', name='Original'))
    
    # Optimized curve
    fig.add_trace(go.Scatter(x=xs.numpy(), y=curves[frame].numpy(), mode='lines', name='Optimized'))
    
    # Optimal flat curve (green line)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Optimal Flat', line=dict(color='green')))
    
    fig.update_layout(
        title='Curve Optimization',
        xaxis_title='X',
        yaxis_title='Y',
        width=800,
        height=600
    )
    
    return fig

# Create a slider for animation control
slider = widgets.IntSlider(
    min=0,
    max=len(curves)-1,
    step=1,
    value=0,
    description='Frame:',
    continuous_update=False
)

# Create a play button
play = widgets.Play(
    min=0,
    max=len(curves)-1,
    step=1,
    interval=50,
    description="Play"
)

# Link the slider and play button
widgets.jslink((play, 'value'), (slider, 'value'))

# Create an Output widget to display the plot
out = widgets.Output()

# Update function for the plot
def update_plot(change):
    with out:
        out.clear_output(wait=True)
        display(create_plot(change['new']))

# Observe slider changes
slider.observe(update_plot, names='value')

# Display the widgets and initial plot
display(widgets.HBox([play, slider]))
display(out)

# Show initial plot
with out:
    display(create_plot(0))

# %%

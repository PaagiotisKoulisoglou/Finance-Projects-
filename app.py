"""
Advanced Black-Scholes Option Calculator
Author: [Your Name]
"""

import streamlit as st
import math 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from datetime import datetime

# Page setup
st.set_page_config(page_title="Options Calculator", layout="wide")

# Simple dark theme
st.markdown("""
    <style>
    .stApp {background-color: #1a1a1a; color: #ffffff;}
    .sidebar {background-color: #2d2d2d;}
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Options Calculator")

# Sidebar inputs with advanced options
with st.sidebar:
    st.header("Parameters")
    
    # Basic Parameters
    S = st.number_input('Asset Price ($)', min_value=1.0, value=100.0)
    K = st.number_input('Strike Price ($)', min_value=1.0, value=100.0)
    T = st.number_input('Time to Expiry (Years)', min_value=0.1, value=1.0)
    r = st.number_input('Risk-Free Rate (%)', min_value=0.0, value=5.0) / 100
    sigma = st.number_input('Volatility (%)', min_value=1.0, value=30.0) / 100

    # Advanced Settings
    st.markdown("---")
    st.subheader("Advanced Settings")
    show_greeks = st.checkbox("Show Greeks", value=True)
    show_probability = st.checkbox("Show Probability Analysis", value=True)

# Black-Scholes calculations
def calculate_bs_prices(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)
    put = K * np.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)
    
    delta_call = scipy.stats.norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = scipy.stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * scipy.stats.norm.pdf(d1) / 100
    theta = (-S * sigma * scipy.stats.norm.pdf(d1) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)) / 365
    
    return call, put, delta_call, delta_put, gamma, vega, theta

# Calculate values
call, put, delta_c, delta_p, gamma, vega, theta = calculate_bs_prices(S, K, T, r, sigma)

# Display prices
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div style='background-color: rgba(0,255,0,0.1); padding: 20px; border-radius: 10px;'><h2 style='color: #4CAF50'>Call: ${call:.2f}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='background-color: rgba(255,0,0,0.1); padding: 20px; border-radius: 10px;'><h2 style='color: #f44336'>Put: ${put:.2f}</h2></div>", unsafe_allow_html=True)

# Display Greeks only if selected
if show_greeks:
    st.markdown("### Greeks")
    greek_cols = st.columns(5)
    with greek_cols[0]: st.metric("Delta Call", f"{delta_c:.3f}")
    with greek_cols[1]: st.metric("Delta Put", f"{delta_p:.3f}")
    with greek_cols[2]: st.metric("Gamma", f"{gamma:.3f}")
    with greek_cols[3]: st.metric("Vega", f"{vega:.3f}")
    with greek_cols[4]: st.metric("Theta", f"{theta:.3f}")

# Show probability analysis if selected
if show_probability:
    st.markdown("### Probability Analysis")
    prob_col1, prob_col2 = st.columns(2)
    
    # Calculate probabilities
    prob_above = 1 - scipy.stats.norm.cdf((np.log(K/S) - (r - 0.5*sigma**2)*T)/(sigma*np.sqrt(T)))
    prob_below = 1 - prob_above
    
    with prob_col1:
        st.metric("Probability Above Strike", f"{prob_above:.1%}")
    with prob_col2:
        st.metric("Probability Below Strike", f"{prob_below:.1%}")

# Heatmaps
spot_range = np.linspace(0.7*K, 1.3*K, 10)
vol_range = np.linspace(0.1, 0.8, 10)[::-1]
X, Y = np.meshgrid(spot_range, vol_range)
Z_call = np.zeros_like(X)
Z_put = np.zeros_like(X)

for i in range(len(vol_range)):
    for j in range(len(spot_range)):
        Z_call[i,j], Z_put[i,j] = calculate_bs_prices(X[i,j], K, T, r, Y[i,j])[:2]

# Plot heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor('#1a1a1a')

def get_text_color(val, min_val, max_val):
    """Determine text color based on background brightness"""
    normalized_val = (val - min_val) / (max_val - min_val)
    return 'black' if normalized_val > 0.5 else 'white'

for ax, Z, title in [(ax1, Z_call, 'CALL'), (ax2, Z_put, 'PUT')]:
    # Create text colors array based on values
    text_colors = [[get_text_color(val, Z.min(), Z.max()) for val in row] for row in Z]
    
    sns.heatmap(Z, 
                xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(vol_range, 2),
                ax=ax,
                cmap='viridis',
                annot=True,
                fmt='.2f',
                annot_kws={'size': 10},
                cbar_kws={'label': 'Price ($)'},
                mask=None,
                center=None,
                robust=True)
    
    # Update text colors after heatmap is created
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            text = ax.texts[i * len(Z[0]) + j]
            text.set_color(text_colors[i][j])
    
    ax.set_title(title, fontsize=16, pad=20, color='white')
    ax.set_xlabel('Spot Price ($)', fontsize=12, color='white')
    ax.set_ylabel('Volatility', fontsize=12, color='white')
    ax.tick_params(colors='white')

plt.tight_layout()
st.pyplot(fig)

# Educational section
st.markdown("""
---
### 📚 Quick Guide

#### How to Use
1. Enter your parameters in the sidebar
2. View option prices at the top
3. Check the Greeks values
4. Explore price variations in the heatmaps

#### Understanding the Greeks
- **Delta (Δ)**: Measures how much the option price changes when the underlying price changes by $1
  - Call Delta ranges from 0 to 1
  - Put Delta ranges from -1 to 0

- **Gamma (Γ)**: Rate of change in Delta for a $1 move in the underlying
  - Same for calls and puts
  - Highest for at-the-money options

- **Vega (ν)**: Measures sensitivity to volatility
  - Shows how much the option price changes for a 1% change in volatility
  - Higher for longer-dated options

- **Theta (Θ)**: Time decay
  - Shows how much value the option loses each day
  - Generally negative for bought options

#### How to Use Heatmaps
- **X-axis (Spot Price)**: Shows how option prices change as the underlying asset price moves.
- **Y-axis (Volatility)**: Shows how option prices change with different volatility levels.
- **Colors**: Brighter colors indicate higher option prices, darker colors indicate lower prices.
- **Numbers**: Each cell shows the exact option price for that specific combination of spot price and volatility.
- **Key Insights**:
  - Call options become more valuable as spot price increases.
  - Put options become more valuable as spot price decreases.
  - Both options generally become more expensive with higher volatility.
  - The relationship between price and volatility is non-linear.
""")

st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


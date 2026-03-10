"""
Cold Chain Monitoring Dashboard
Interactive Streamlit Dashboard with Route Animation and Real-Time Metrics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import random
from dataclasses import asdict
import time

from config import SimConfig
from synthetic_data import build_hub_to_city_instance
from vrptw_solver import solve_vrptw
from sim_engine import simulate_routes
from monitoring import estimate_quality_remaining
from real_geography import REAL_GEOGRAPHY

# Page configuration
st.set_page_config(
    page_title="Cold Chain Monitoring",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .danger-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)



def create_route_animation(inst, sim_res):
    """Create animated map with city labels, arrows, and moving vehicles"""
    
    # GPS coordinates and city names
    locations = {}
    for node_id, (lat, lon) in inst.coords.items():
        locations[node_id] = (lat, lon)
    
    city_names = {}
    city_names[inst.start] = REAL_GEOGRAPHY["hub"]["name"]
    city_names[inst.end] = REAL_GEOGRAPHY["hub"]["name"]
    for customer_data in REAL_GEOGRAPHY["customers"]:
        city_names[customer_data["id"]] = customer_data["name"]
    
    # Build animation frames
    frames = []
    
    # Safety check: if no routes were generated, return empty figure
    if not sim_res.final_states or len(sim_res.final_states) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ No feasible solution found. Try adjusting weights or parameters.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    max_time = max(state.clock_min for state in sim_res.final_states.values())
    time_steps = list(range(0, int(max_time) + 1, 5))  # 5-minute intervals
    
    colors = ['#667eea', '#f2994a', '#56ab2f']
    
    for t in time_steps:
        frame_data = []
        
        # Static customer markers (blue dots)
        customer_lats = [locations[c][0] for c in inst.customers]
        customer_lons = [locations[c][1] for c in inst.customers]
        
        frame_data.append(go.Scattergeo(
            lat=customer_lats,
            lon=customer_lons,
            mode='markers+text',
            marker=dict(size=10, color='#4169E1', line=dict(width=1, color='white')),
            text=[city_names.get(c, f'C{c}') for c in inst.customers],
            textposition='top center',
            textfont=dict(size=9, color='darkblue', family='Arial Black'),
            name='Customer Cities',
            showlegend=False,
            hovertext=[city_names.get(c, f'Customer {c}') for c in inst.customers],
            hoverinfo='text'
        ))
        
        # Static depot (red star with label)
        depot_lat, depot_lon = locations[inst.start]
        frame_data.append(go.Scattergeo(
            lat=[depot_lat],
            lon=[depot_lon],
            mode='markers+text',
            marker=dict(size=18, color='red', symbol='star', line=dict(width=2, color='darkred')),
            text=city_names.get(inst.start, 'Midnapore'),
            textposition='bottom center',
            textfont=dict(size=11, color='darkred', family='Arial Black'),
            name=f'{city_names.get(inst.start, "Midnapore")} (Hub)',
            showlegend=False,
            hovertext=city_names.get(inst.start, 'HUB'),
            hoverinfo='text'
        ))
        
        # Vehicle routes and positions
        for k, state in sim_res.final_states.items():
            route = state.route
            route_lats = []
            route_lons = []
            
            for node in route:
                if node in locations:
                    route_lats.append(locations[node][0])
                    route_lons.append(locations[node][1])
            
            if len(route_lats) >= 2:
                # Draw route line
                frame_data.append(go.Scattergeo(
                    lat=route_lats,
                    lon=route_lons,
                    mode='lines',
                    line=dict(width=3, color=colors[k % len(colors)]),
                    name=f'Vehicle {k} Route',
                    showlegend=(t == 0),  # Show legend only on first frame
                    hovertemplate=f'<b>Vehicle {k}</b><extra></extra>'
                ))
                
                # Calculate vehicle position at time t
                progress = min(t / max_time, 1.0) if max_time > 0 else 0
                segment_count = len(route_lats) - 1
                current_segment = int(progress * segment_count)
                
                if current_segment < segment_count:
                    # Interpolate within current segment
                    segment_progress = (progress * segment_count) - current_segment
                    
                    lat1 = route_lats[current_segment]
                    lon1 = route_lons[current_segment]
                    lat2 = route_lats[current_segment + 1]
                    lon2 = route_lons[current_segment + 1]
                    
                    vehicle_lat = lat1 + (lat2 - lat1) * segment_progress
                    vehicle_lon = lon1 + (lon2 - lon1) * segment_progress
                else:
                    # At end of route
                    vehicle_lat = route_lats[-1]
                    vehicle_lon = route_lons[-1]
                
                # Add moving vehicle marker with label
                frame_data.append(go.Scattergeo(
                    lat=[vehicle_lat],
                    lon=[vehicle_lon],
                    mode='markers+text',
                    marker=dict(
                        size=14,
                        color=colors[k % len(colors)],
                        symbol='circle',
                        opacity=0.9
                    ),
                    text=f'V{k}',
                    textposition='middle center',
                    textfont=dict(size=8, color='white', family='Arial Black'),
                    name=f'Vehicle {k}',
                    showlegend=False,
                    hovertemplate=f'<b>Vehicle {k}</b><br>Time: {t} min<extra></extra>'
                ))
        
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    # Create figure with first frame
    fig = go.Figure(data=frames[0].data, frames=frames)
    
    # Configure mapbox
    center_lat = REAL_GEOGRAPHY["hub"]["latitude"]
    center_lon = REAL_GEOGRAPHY["hub"]["longitude"]
    
    fig.update_layout(
        geo=dict(
            scope='asia',
            projection_type='mercator',
            showland=True,
            landcolor='rgb(230, 230, 230)',
            showcountries=True,
            countrycolor='rgb(150, 150, 150)',
            showlakes=True,
            lakecolor='rgb(180, 220, 255)',
            center=dict(lat=center_lat, lon=center_lon),
            lonaxis=dict(range=[center_lon - 1.5, center_lon + 1.5]),
            lataxis=dict(range=[center_lat - 1, center_lat + 1]),
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=11)
        ),
        height=650,
        title=dict(
            text="🚚 Cold Chain Routes - Medinipur District",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#667eea')
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white",  # Ensure white background
        # Animation controls
        updatemenus=[{
            "buttons": [
                {
                    "label": "▶️ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True,
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }]
                },
                {
                    "label": "⏸️ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate"
                    }]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "left",
            "y": 0,
            "yanchor": "bottom",
            "bgcolor": "#667eea",
            "font": {"color": "white", "size": 13}
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "args": [[f.name], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate"
                    }],
                    "label": f"{f.name}m",
                    "method": "animate"
                }
                for f in frames
            ],
            "x": 0.1,
            "len": 0.85,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top",
            "pad": {"b": 10, "t": 50},
            "currentvalue": {
                "prefix": "Time: ",
                "suffix": " min",
                "visible": True,
                "xanchor": "right"
            }
        }]
    )
    
    # Hide the coordinate axes (remove overlay)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    return fig


def create_quality_timeline(inst, sim_res):
    """Create interactive quality degradation timeline"""
    
    fig = make_subplots(
        rows=len(sim_res.final_states), cols=1,
        subplot_titles=[f'Vehicle {k}' for k in sorted(sim_res.final_states.keys())],
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx, (k, state) in enumerate(sorted(sim_res.final_states.items())):
        if state.batch_metrics:
            for batch_id, metrics in state.batch_metrics.items():
                batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                if batch:
                    # Simple quality timeline (could be enhanced with actual time series)
                    time_points = [0, metrics.above_safe_minutes / 2, metrics.above_safe_minutes]
                    quality_points = [
                        1.0,
                        estimate_quality_remaining(batch, metrics) + 0.15,
                        estimate_quality_remaining(batch, metrics)
                    ]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=quality_points,
                            mode='lines+markers',
                            name=f'{batch.produce_type}',
                            line=dict(color=colors[batch_id % len(colors)], width=3),
                            marker=dict(size=8),
                            hovertemplate='Time: %{x:.0f} min<br>Quality: %{y:.1%}<extra></extra>'
                        ),
                        row=idx+1, col=1
                    )
        
        # Add threshold line
        fig.add_hline(
            y=0.6, line_dash="dash", line_color="red",
            annotation_text="Min Quality", annotation_position="right",
            row=idx+1, col=1
        )
        
        fig.update_yaxes(title_text="Quality %", row=idx+1, col=1, range=[0, 1.05])
        fig.update_xaxes(title_text="Time (min)", row=idx+1, col=1)
    
    fig.update_layout(
        height=300 * len(sim_res.final_states),
        title_text="Product Quality Degradation Over Time",
        showlegend=True
    )
    
    return fig


def create_temperature_heatmap(inst, sim_res):
    """Create temperature heatmap for all compartments"""
    
    # Collect temperature data
    temp_data = []
    for k, state in sim_res.final_states.items():
        for comp_name, temp in state.compartment_temps.items():
            vehicle_meta = inst.vehicle_meta[k]
            setpoint = vehicle_meta.compartments[comp_name].setpoint_c
            temp_data.append({
                'Vehicle': f'V{k}',
                'Compartment': comp_name,
                'Temperature': temp,
                'Setpoint': setpoint,
                'Deviation': temp - setpoint
            })
    
    df = pd.DataFrame(temp_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=df['Compartment'],
        y=df['Vehicle'],
        z=df['Deviation'],
        colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
        zmid=0,
        text=df['Temperature'].round(1),
        texttemplate='%{text}°C',
        hovertemplate='Vehicle: %{y}<br>Compartment: %{x}<br>Temp: %{text}°C<extra></extra>',
        colorbar=dict(title="Temp Deviation (°C)")
    ))
    
    fig.update_layout(
        title='Compartment Temperature Status',
        xaxis_title='Compartment',
        yaxis_title='Vehicle',
        height=250
    )
    
    return fig


def display_reroute_timeline(sim_res):
    """Display reroute decisions as interactive timeline"""
    
    reroute_events = [ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED"]
    
    if not reroute_events:
        st.info("ℹ️ No rerouting decisions made during simulation")
        return
    
    # Create timeline data
    timeline_data = []
    for ev in reroute_events:
        timeline_data.append({
            'Time': ev.t_min,
            'Vehicle': f'Vehicle {ev.vehicle_id}',
            'Reason': ev.details.get('reason', 'Unknown'),
            'Decision': ev.details.get('option_selected', 'N/A'),
            'Score': ev.details.get('score', 0)
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Create timeline figure
    fig = px.scatter(df, x='Time', y='Vehicle', color='Reason', size='Score',
                     hover_data=['Decision', 'Reason', 'Score'],
                     title='Rerouting Decisions Timeline',
                     labels={'Time': 'Simulation Time (min)'})
    
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show details in expandable sections
    for i, event in enumerate(reroute_events):
        with st.expander(f"🔄 Decision #{i+1} - Time: {event.t_min:.1f} min - Vehicle {event.vehicle_id}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Trigger Reason", event.details.get('reason', 'N/A'))
            col2.metric("Option Chosen", event.details.get('option_selected', 'N/A'))
            col3.metric("Decision Score", f"{event.details.get('score', 0):.2f}")


def main():
    # Header
    st.markdown('<div class="main-header">🚚 Cold Chain Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    padding: 1rem; border-radius: 10px; text-align: center;
                    color: white; font-size: 1.2rem; font-weight: bold;
                    letter-spacing: 1px; margin-bottom: 0.5rem;">
            🚚 Cold Chain AI
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### ⚙️ Simulation Configuration")
        
        # Fixed number of customers (12 real cities in Medinipur)
        n_customers = 12
        st.info(f"📍 **{n_customers} Customer Cities** (Medinipur District)")
        
        n_vehicles = st.slider("Number of Vehicles", 2, 5, 3)
        vehicle_capacity = st.slider("Vehicle Capacity (units)", 10, 30, 20)
        
        st.markdown("### 📦 Customer Demands")
        st.caption("Set demand for each city (crates/units)")
        
        # Individual demand sliders for each customer
        customer_demands = {}
        from real_geography import REAL_GEOGRAPHY
        
        # Create 2 columns for better layout
        col1, col2 = st.columns(2)
        
        for idx, customer in enumerate(REAL_GEOGRAPHY["customers"]):
            city_name = customer["name"]
            customer_id = customer["id"]
            
            # Alternate between columns
            with col1 if idx % 2 == 0 else col2:
                customer_demands[customer_id] = st.number_input(
                    f"{city_name}",
                    min_value=1,
                    max_value=10,
                    value=random.randint(1, 4),
                    step=1,
                    key=f"demand_{customer_id}"
                )
        
        st.markdown("### 🎯 Quality Thresholds")
        min_quality = st.slider("Minimum Quality (%)", 40, 90, 60) / 100
        
        st.markdown("### ⚖️ VRPTW Optimization Weights")
        st.caption("Economic costs: Distance (₹/km), Time (₹/min), Risk (₹/unit)")
        alpha_weight = st.slider("Distance Cost α (₹/km)", 5.0, 20.0, 12.0, 0.5)
        beta_weight = st.slider("Time Cost β (₹/min)", 2.0, 10.0, 5.0, 0.5)
        gamma_weight = st.slider("Risk Cost γ (₹/unit)", 10.0, 50.0, 30.0, 1.0)
        
        st.markdown("### 💰 Economic Parameters")
        
        # Revenue
        st.markdown("**💵 Revenue**")
        revenue_per_customer = st.number_input("Revenue per Customer (₹)", 500, 2000, 1000, key="revenue")
        
        # Operating Costs
        st.markdown("**� Operating Costs**")
        col1, col2 = st.columns(2)
        with col1:
            fuel_cost_per_km = st.number_input("Fuel (₹/km)", 5, 20, 12, key="fuel")
        with col2:
            driver_wage_per_hour = st.number_input("Driver (₹/hr)", 50, 200, 100, key="driver")
        
        col3, col4 = st.columns(2)
        with col3:
            vehicle_rental_per_day = st.number_input("Vehicle Rental (₹/day)", 1000, 5000, 2000, key="rental")
        with col4:
            refrigeration_cost_per_hour = st.number_input("Refrigeration (₹/hr)", 20, 100, 50, key="refrig")
        
        # Quality & Penalty Costs
        st.markdown("**⚠️ Quality & Penalties**")
        col5, col6 = st.columns(2)
        with col5:
            spoilage_cost = st.number_input("Spoilage (₹/unit)", 1000, 5000, 2700, key="spoilage")
        with col6:
            temp_violation_penalty = st.number_input("Temp Violation (₹)", 100, 1000, 500, key="temp_penalty")
        
        st.markdown("---")
        run_button = st.button("🚀 Run Simulation", use_container_width=True)
    
    # Main content
    if run_button or 'simulation_run' not in st.session_state:
        with st.spinner("🔄 Running simulation..."):
            # Create config with custom parameters using replace
            from dataclasses import replace
            base_cfg = SimConfig()
            cfg = replace(
                base_cfg,
                trigger_min_quality=min_quality,
                revenue_per_customer=revenue_per_customer,
                spoilage_cost_per_unit=spoilage_cost
            )
            
            # Run simulation
            inst = build_hub_to_city_instance(
                seed=7,
                n_city_points=n_customers,
                n_vehicles=n_vehicles,
                vehicle_capacity=vehicle_capacity,
                horizon_min=cfg.horizon_min,
                cfg=cfg,
                custom_demands=customer_demands  # Pass custom demands
            )
            
            res = solve_vrptw(inst, alpha=alpha_weight, beta=beta_weight, gamma=gamma_weight, verbose=False)
            sim_res = simulate_routes(inst, res.routes, cfg)
            
            # Store in session state
            st.session_state['simulation_run'] = True
            st.session_state['inst'] = inst
            st.session_state['res'] = res
            st.session_state['sim_res'] = sim_res
            st.session_state['cfg'] = cfg
    
    if 'simulation_run' in st.session_state:
        inst = st.session_state['inst']
        res = st.session_state['res']
        sim_res = st.session_state['sim_res']
        cfg = st.session_state['cfg']
        
        # Key Metrics Row
        st.markdown("### 📊 Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        
        # Calculate metrics
        fulfilled = sum(1 for ev in sim_res.events if ev.event == "SERVICE_START" and ev.details.get('node') in inst.customers)
        total_customers = len(inst.customers)
        fulfillment_rate = (fulfilled / total_customers * 100) if total_customers > 0 else 0
        
        # Average quality
        all_qualities = []
        for state in sim_res.final_states.values():
            if state.batch_metrics:
                for batch_id, metrics in state.batch_metrics.items():
                    batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                    if batch:
                        all_qualities.append(estimate_quality_remaining(batch, metrics))
        avg_quality = (sum(all_qualities) / len(all_qualities) * 100) if all_qualities else 0
        
        # Reroute count
        reroute_count = len([ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED"])
        
        # Total distance
        total_dist = res.objective_value if hasattr(res, 'objective_value') else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Fulfillment</h3>
                <h1>{fulfillment_rate:.1f}%</h1>
                <p>{fulfilled}/{total_customers} customers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            quality_card_class = "success-card" if avg_quality >= 80 else "warning-card" if avg_quality >= 60 else "danger-card"
            st.markdown(f"""
            <div class="{quality_card_class}">
                <h3>✨ Avg Quality</h3>
                <h1>{avg_quality:.1f}%</h1>
                <p>All deliveries</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔄 Reroutes</h3>
                <h1>{reroute_count}</h1>
                <p>Decisions made</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Route Animation Section
        st.markdown("### 🗺️ Live Route Visualization")
        route_fig = create_route_animation(inst, sim_res)
        st.plotly_chart(route_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Reroute Timeline
        st.markdown("### 🔄 Rerouting Decisions")
        display_reroute_timeline(sim_res)
        
        st.markdown("---")
        
        # Two column layout for graphs
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📈 Quality Degradation")
            quality_fig = create_quality_timeline(inst, sim_res)
            st.plotly_chart(quality_fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 🌡️ Temperature Status")
            temp_fig = create_temperature_heatmap(inst, sim_res)
            st.plotly_chart(temp_fig, use_container_width=True)
            
            st.markdown("### 📋 Vehicle Summary")
            for k, state in sorted(sim_res.final_states.items()):
                # Calculate distance and time for this vehicle
                route = state.route
                vehicle_dist = 0.0
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    vehicle_dist += inst.dist.get((from_node, to_node), 0.0)
                
                vehicle_time = state.clock_min
                
                with st.expander(f"🚚 Vehicle {k} - {vehicle_dist:.1f} km, {vehicle_time:.0f} min"):
                    # Top metrics
                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Distance", f"{vehicle_dist:.1f} km")
                    met2.metric("Time", f"{vehicle_time:.0f} min")
                    met3.metric("Delayed", f"{state.delayed_min:.1f} min")
                    rerouted = "✅ Yes" if state.reroute_triggered else "❌ No"
                    met4.metric("Rerouted", rerouted)
                    
                    st.write("**Route:**", " → ".join(str(n) for n in route))
                    
                    st.write("**Compartment Temperatures:**")
                    for comp, temp in state.compartment_temps.items():
                        setpoint = inst.vehicle_meta[k].compartments[comp].setpoint_c
                        deviation = temp - setpoint
                        color = "🔴" if abs(deviation) > 3 else "🟡" if abs(deviation) > 1 else "🟢"
                        st.write(f"{color} {comp}: {temp:.2f}°C (setpoint: {setpoint}°C, deviation: {deviation:+.2f}°C)")
        
        st.markdown("---")
        
        # Customer Fulfillment Table
        st.markdown("### 👥 Customer Fulfillment Details")
        
        customer_data = []
        for customer in inst.customers:
            # Check if served
            served = any(ev.event == "SERVICE_START" and ev.details.get('node') == customer for ev in sim_res.events)
            
            # Get shipment info
            shipment = next((s for s in inst.shipments if s.customer_node_id == customer), None)
            if shipment:
                batch = shipment.batch
                product = batch.produce_type
                units = shipment.demand_units
                
                # Get quality if served
                if served:
                    for state in sim_res.final_states.values():
                        if batch.batch_id in state.batch_metrics:
                            quality = estimate_quality_remaining(batch, state.batch_metrics[batch.batch_id])
                            break
                    else:
                        quality = 0
                else:
                    quality = None
                
                customer_data.append({
                    'Customer': f'C{customer}',
                    'Product': product,
                    'Units': units,
                    'Status': '✅ Delivered' if served else '❌ Not Delivered',
                    'Quality': f'{quality:.1%}' if quality is not None else 'N/A'
                })
        
        df_customers = pd.DataFrame(customer_data)
        st.dataframe(df_customers, use_container_width=True, hide_index=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>👋 Welcome to the Cold Chain Monitoring System</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Configure your simulation parameters in the sidebar and click <strong>Run Simulation</strong> to begin.
            </p>
            <p style="margin-top: 2rem; font-size: 1rem; color: #888;">
                This dashboard provides real-time monitoring of cold chain logistics with:<br>
                🗺️ Animated route visualization | 📊 Quality tracking | 🔄 Autonomous rerouting | 🌡️ Temperature monitoring
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

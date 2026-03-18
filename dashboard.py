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
import requests
import folium
from streamlit_folium import st_folium
import io
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from config import SimConfig
from synthetic_data import build_hub_to_city_instance
from vrptw_solver import solve_vrptw
from sim_engine import simulate_routes
from monitoring import estimate_quality_remaining
from real_geography import REAL_GEOGRAPHY

def plotly_to_image(fig, render_width=800, render_height=500, pdf_width=None, pdf_height=None):
    """Converts a plotly Figure to a ReportLab Image object using Kaleido."""
    img_bytes = fig.to_image(format="png", width=render_width, height=render_height, scale=2)
    img_io = io.BytesIO(img_bytes)
    
    if pdf_width is None:
        pdf_width = render_width * 0.8
    if pdf_height is None:
        pdf_height = render_height * 0.8
        
    return RLImage(img_io, width=pdf_width, height=pdf_height)

def generate_pdf_report(inst, sim_res, res, df_customers, fulfillment_rate, avg_quality, reroute_count):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=20, leftMargin=20,
                            topMargin=20, bottomMargin=15)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1, fontSize=16, spaceAfter=8)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=6, textColor=colors.HexColor('#667eea'))
    h3_style = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11, spaceAfter=4, textColor=colors.HexColor('#444444'))
    normal_style = styles['Normal']
    
    elements = []
    
    elements.append(Paragraph("Cold Chain Monitoring - Simulation Report", title_style))
    elements.append(Spacer(1, 5))
    
    # 1. Global KPIs
    elements.append(Paragraph("Global Key Performance Indicators", h2_style))
    kpi_data = [
        ["Fulfillment Rate", f"{fulfillment_rate:.1f}%"],
        ["Average Quality Delivered", f"{avg_quality:.1f}%"],
        ["Total Autonomous Reroutes", str(reroute_count)]
    ]
    kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 10))
    
    # 2. Vehicle Sub-reports
    for idx, k in enumerate(sorted(sim_res.final_states.keys())):
        if idx > 0:
            elements.append(Spacer(1, 20))
            # Add a subtle separator between vehicle details
            sep = Table([[""]], colWidths=[7.5*inch])
            sep.setStyle(TableStyle([('LINEABOVE', (0,0), (-1,-1), 1, colors.HexColor('#e0e0e0'))]))
            elements.append(sep)
            elements.append(Spacer(1, 15))
            
        elements.append(Paragraph(f"Vehicle {k+1} Detailed Report", title_style))
        
        state = sim_res.final_states.get(k)
        route = state.route
        vehicle_dist = sum(inst.dist.get((route[i], route[i+1]), 0.0) for i in range(len(route) - 1))
        
        route_str = " -> ".join("Depot" if n in (inst.start, inst.end) else (inst.node_meta[n].name if hasattr(inst, 'node_meta') and n in inst.node_meta else str(n)) for n in route)
        
        elements.append(Paragraph("Route Overview", h2_style))
        elements.append(Paragraph(f"<b>Path:</b> {route_str}", normal_style))
        elements.append(Paragraph(f"<b>Total Distance:</b> {vehicle_dist:.1f} km &nbsp;&nbsp;|&nbsp;&nbsp; <b>Total Duration:</b> {state.clock_min:.0f} min", normal_style))
        elements.append(Spacer(1, 8))
        
        # Compartment Temperatures Table
        elements.append(Paragraph("Final Compartment Temperatures", h3_style))
        temps_data = [["Compartment", "Temperature", "Setpoint", "Deviation"]]
        veh_meta = inst.vehicle_meta.get(k)
        for comp, temp in state.compartment_temps.items():
            setpoint = veh_meta.compartments[comp].setpoint_c if veh_meta else 0
            dev = temp - setpoint
            temps_data.append([f"Comp {comp}", f"{temp:.2f} C", f"{setpoint:.1f} C", f"{dev:+.2f} C"])
        
        t = Table(temps_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 8))
        
        # Vehicle Specific Graphs (Quality + Temperature) - Side by Side
        q_fig = create_quality_figure_for_vehicle(inst, sim_res, k)
        q_fig.update_layout(title="", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black", size=16), margin=dict(t=25, b=25, l=25, r=25))
        q_fig.update_xaxes(gridcolor="lightgrey")
        q_fig.update_yaxes(gridcolor="lightgrey")
        
        t_fig = create_temperature_figure_for_vehicle(inst, sim_res, k)
        t_fig.update_layout(title="", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black", size=16), margin=dict(t=25, b=25, l=25, r=25))
        t_fig.update_xaxes(gridcolor="lightgrey")
        t_fig.update_yaxes(gridcolor="lightgrey")
        
        try:
            # Render at a large internal resolution (800x480) so Plotly elements aren't cramped
            # and then shrink down to fit nicely side-by-side in the PDF (3.6x2.16 inches)
            img_q = plotly_to_image(q_fig, render_width=800, render_height=480, pdf_width=3.6*inch, pdf_height=2.16*inch)
            img_t = plotly_to_image(t_fig, render_width=800, render_height=480, pdf_width=3.6*inch, pdf_height=2.16*inch)
            
            graph_data = [
                [Paragraph("Quality Degradation Over Time", h3_style), Paragraph("Temperature Monitoring Over Time", h3_style)],
                [img_q, img_t]
            ]
            graph_table = Table(graph_data, colWidths=[3.7*inch, 3.7*inch])
            graph_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ]))
            elements.append(graph_table)
        except Exception as e:
            elements.append(Paragraph(f"<i>Could not render charts: {e}</i>", normal_style))
        elements.append(Spacer(1, 12))
        
        # Fulfillment Data
        veh_customers = df_customers[df_customers['Vehicle'] == f"Vehicle {k+1}"]
        if not veh_customers.empty:
            elements.append(Paragraph("Customer Fulfillment Log", h3_style))
            cust_data = [["Customer", "Product", "Units", "Status", "Final Quality", "Failure Reason"]]
            for _, row in veh_customers.iterrows():
                cust_data.append([
                    str(row['Customer']), str(row['Product']), str(row['Units']),
                    str(row['Status']), str(row['Final Quality Delivered']), str(row['Failure Reason'])
                ])
            
            c_table = Table(cust_data, colWidths=[1.5*inch, 1*inch, 0.8*inch, 1.2*inch, 1*inch, 1.5*inch])
            c_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
            ]))
            elements.append(c_table)
        
    # Global Reroute Decisions
    reroute_events = [ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED"]
    if reroute_events:
        elements.append(Spacer(1, 20))
        sep = Table([[""]], colWidths=[7.5*inch])
        sep.setStyle(TableStyle([('LINEABOVE', (0,0), (-1,-1), 1, colors.HexColor('#eb3349'))]))
        elements.append(sep)
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Global Rerouting Events", title_style))
        r_data = [["Time (min)", "Vehicle", "Reason", "Action"]]
        for ev in reroute_events:
            r_data.append([
                f"{ev.t_min:.1f}", f"Vehicle {ev.vehicle_id+1}", 
                str(ev.details.get('reason')), str(ev.details.get('option_selected'))
            ])
        r_table = Table(r_data, colWidths=[1*inch, 1*inch, 2.5*inch, 2.5*inch])
        r_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eb3349')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(r_table)
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

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



@st.cache_data(show_spinner=False)
def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
    """Fetch real road polyline from OSRM between two (lat, lon) points.
    Cached per unique city pair so API is only called once per session.
    Falls back to straight-line if OSRM is unreachable.
    """
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson"
    )
    try:
        resp = requests.get(url, timeout=6)
        data = resp.json()
        if data.get("code") == "Ok":
            coords = data["routes"][0]["geometry"]["coordinates"]
            # OSRM returns [lon, lat] — we need (lat, lon)
            return [(lat, lon) for lon, lat in coords]
    except Exception:
        pass
    # Fallback to straight line
    return [(start_lat, start_lon), (end_lat, end_lon)]


def build_vehicle_road_timeline(inst, sim_res):
    """Build per-vehicle list of (depart_time, arrive_time, road_poly) tuples for each segment.
    Uses the detailed 5-min log_rows to trace exactly when a vehicle leaves a node
    and arrives at the next.
    """
    locations = {}
    for node_id, (lat, lon) in inst.coords.items():
        locations[node_id] = (lat, lon)

    timelines = {}

    for k, state in sim_res.final_states.items():
        route = state.route
        segments = []
        
        # We need to find depart_t and arrive_t for every segment `(route[i], route[i+1])`
        # We trace through log_rows for vehicle `k`.
        v_logs = [r for r in sim_res.log_rows if r['vehicle'] == k]
        
        if not v_logs:
            continue
            
        seg_idx = 0
        current_depart_t = 0.0
        
        # A vehicle is 'traveling' if its remaining_dist_to_next < total_arc_dist, but > 0
        # A vehicle 'arrives' when current_node switches to the next node.
        
        for r_idx in range(len(v_logs) - 1):
            if seg_idx >= len(route) - 1:
                break
                
            curr_log = v_logs[r_idx]
            next_log = v_logs[r_idx + 1]
            
            from_node = route[seg_idx]
            to_node = route[seg_idx + 1]
            
            # Did the vehicle complete the segment between curr_log and next_log?
            # It finishes a segment when its 'current_node' updates to the 'to_node'
            if curr_log['current_node'] == from_node and next_log['current_node'] == to_node:
                arrive_t = next_log['t_min']
                
                if from_node in locations and to_node in locations:
                    f_lat, f_lon = locations[from_node]
                    t_lat, t_lon = locations[to_node]
                    road_poly = get_osrm_route(f_lat, f_lon, t_lat, t_lon)
                    segments.append((current_depart_t, arrive_t, f_lat, f_lon, t_lat, t_lon, road_poly))
                
                # The next segment starts departing after the service time at `to_node`
                service_min = inst.service.get(to_node, 0.0) if to_node in inst.customers else 0.0
                current_depart_t = arrive_t + service_min
                seg_idx += 1
                
        # If it finished the last segment right at the end of the logs
        if seg_idx < len(route) - 1:
            from_node = route[seg_idx]
            to_node = route[seg_idx + 1]
            if from_node in locations and to_node in locations:
                f_lat, f_lon = locations[from_node]
                t_lat, t_lon = locations[to_node]
                road_poly = get_osrm_route(f_lat, f_lon, t_lat, t_lon)
                # extrapolate arrival based on clock_min
                arrive_t = state.clock_min
                segments.append((current_depart_t, arrive_t, f_lat, f_lon, t_lat, t_lon, road_poly))

        timelines[k] = segments

    return timelines


def create_live_folium_map(inst, sim_res, current_sim_time):
    """Render a Folium map with vehicles following real roads,
    positioned accurately at current_sim_time relative to simulation."""
    locations = {}
    for node_id, (lat, lon) in inst.coords.items():
        locations[node_id] = (lat, lon)

    city_names = {}
    city_names[inst.start] = REAL_GEOGRAPHY["hub"]["name"]
    city_names[inst.end] = REAL_GEOGRAPHY["hub"]["name"]
    for c in REAL_GEOGRAPHY["customers"]:
        city_names[c["id"]] = c["name"]

    center_lat = REAL_GEOGRAPHY["hub"]["latitude"]
    center_lon = REAL_GEOGRAPHY["hub"]["longitude"]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9,
                   tiles="OpenStreetMap")

    # ── Customer markers with permanent labels ────────────────────────────────
    for c in inst.customers:
        lat, lon = locations[c]
        cname = city_names.get(c, f'C{c}')
        folium.Marker(
            [lat, lon],
            tooltip=cname,
            popup=cname,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # ── Depot marker ─────────────────────────────────────────────────────────
    hub_lat, hub_lon = locations[inst.start]
    hub_name = city_names.get(inst.start, 'Hub')
    folium.Marker(
        [hub_lat, hub_lon],
        tooltip=f"Hub: {hub_name}",
        popup=f"Hub: {hub_name}",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m)

    # ── Per-vehicle road routes + animated vehicle position ───────────────────
    veh_colors = ['blue', 'orange', 'green', 'purple', 'red', 'cadetblue']
    timelines = build_vehicle_road_timeline(inst, sim_res)

    for k, segments in timelines.items():
        color = veh_colors[k % len(veh_colors)]
        vehicle_lat = vehicle_lon = None

        for depart_t, arrive_t, f_lat, f_lon, t_lat, t_lon, road_poly in segments:
            # Draw road segment
            folium.PolyLine(
                road_poly,
                color=color,
                weight=4,
                opacity=0.8,
                tooltip=f'Vehicle {k} Route'
            ).add_to(m)

            # Is the vehicle currently in this segment?
            if depart_t <= current_sim_time <= arrive_t:
                span = arrive_t - depart_t
                frac = (current_sim_time - depart_t) / span if span > 0 else 0.0
                poly_idx = int(frac * (len(road_poly) - 1))
                poly_idx = max(0, min(poly_idx, len(road_poly) - 1))
                vehicle_lat, vehicle_lon = road_poly[poly_idx]

        # If simulation hasn't started yet, show at depot
        if vehicle_lat is None and current_sim_time == 0:
            vehicle_lat, vehicle_lon = hub_lat, hub_lon
        # If finished, show at last waypoint of last segment
        elif vehicle_lat is None and segments:
            vehicle_lat, vehicle_lon = segments[-1][6][-1]

        if vehicle_lat is not None:
            folium.Marker(
                [vehicle_lat, vehicle_lon],
                tooltip=f'Vehicle {k} (Time: {current_sim_time:.1f}m)',
                popup=f'Vehicle {k}',
                icon=folium.Icon(color=color, icon='truck', prefix='fa')
            ).add_to(m)

    return m


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
        
        frame_data.append(go.Scattermapbox(
            lat=customer_lats,
            lon=customer_lons,
            mode='markers+text',
            marker=dict(size=10, color='#4169E1'),
            text=[city_names.get(c, f'C{c}') for c in inst.customers],
            textposition='bottom right',
            textfont=dict(size=14, color='black', family='Arial Black'),
            name='Customer Cities',
            showlegend=False,
            hoverinfo='none', # Let the text just sit there without interactive hide
        ))
        
        # Static depot (red star with label)
        depot_lat, depot_lon = locations[inst.start]
        frame_data.append(go.Scattermapbox(
            lat=[depot_lat],
            lon=[depot_lon],
            mode='markers+text',
            marker=dict(size=18, color='red'),
            text=[city_names.get(inst.start, 'Midnapore')],
            textposition='bottom center',
            textfont=dict(size=14, color='darkred', family='Arial Black'),
            name=f'{city_names.get(inst.start, "Midnapore")} (Hub)',
            showlegend=False,
            hovertext=[city_names.get(inst.start, 'HUB')],
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
                frame_data.append(go.Scattermapbox(
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
                frame_data.append(go.Scattermapbox(
                    lat=[vehicle_lat],
                    lon=[vehicle_lon],
                    mode='markers+text',
                    marker=dict(
                        size=14,
                        color=colors[k % len(colors)],
                        opacity=0.9
                    ),
                    text=[f'V{k}'],
                    textposition='middle center',
                    textfont=dict(size=10, color='white', family='Arial Black'),
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
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8.5
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


def display_reroute_timeline(sim_res, inst, res, selected_vehicle=None):
    """Display reroute decisions as interactive timeline"""
    
    reroute_events = [ev for ev in sim_res.events if ev.event == "REROUTE_APPLIED"]
    if selected_vehicle is not None:
        reroute_events = [ev for ev in reroute_events if ev.vehicle_id == selected_vehicle]
    
    if not reroute_events:
        st.info("ℹ️ No rerouting decisions made during simulation")
        return
    
    # Dictionaries for user-friendly labels
    reason_map = {
        "TEMP_EXCURSION": "Extended Temperature Excursion",
        "CUMULATIVE_ABUSE": "High Total Temperature Abuse",
        "LOW_QUALITY": "Critical Quality Drop",
        "PREDICTED_LATE": "Predicted Late Arrival"
    }
    
    decision_map = {
        "skip_customer": "Skip Customer Location",
        "swap_order": "Swap Delivery Order",
        "return_to_hub": "Return to Hub Immediately"
    }
    
    # Create timeline data
    timeline_data = []
    for ev in reroute_events:
        raw_reason = ev.details.get('reason', 'Unknown')
        raw_decision = ev.details.get('option_selected', 'N/A')
        
        timeline_data.append({
            'Time': ev.t_min,
            'Vehicle': f'Vehicle {ev.vehicle_id}',
            'Reason': reason_map.get(raw_reason, raw_reason.replace("_", " ").title()),
            'Decision': decision_map.get(raw_decision, raw_decision.replace("_", " ").title()),
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
            col1, col2 = st.columns(2)
            
            raw_reason = event.details.get('reason', 'N/A')
            raw_decision = event.details.get('option_selected', 'N/A')
            
            friendly_reason = reason_map.get(raw_reason, raw_reason.replace("_", " ").title())
            friendly_decision = decision_map.get(raw_decision, raw_decision.replace("_", " ").title())
            
            col1.markdown(f"<div style='font-size:0.75rem;color:#aaa'>Trigger Reason</div><div style='font-size:1rem;font-weight:600'>{friendly_reason}</div>", unsafe_allow_html=True)
            col2.markdown(f"<div style='font-size:0.75rem;color:#aaa'>Option Chosen</div><div style='font-size:1rem;font-weight:600'>{friendly_decision}</div>", unsafe_allow_html=True)

            # If skip_customer, show exactly which customer is being skipped
            if raw_decision == "skip_customer":
                option_name = event.details.get('option_name', '')
                skipped_node = None
                try:
                    node_str = option_name.split("Skip customer ")[1].split(" ")[0]
                    skipped_node = int(node_str)
                except:
                    pass
                
                if skipped_node is not None:
                    if hasattr(inst, 'node_meta') and skipped_node in inst.node_meta:
                        skipped_name = inst.node_meta[skipped_node].name
                    else:
                        from real_geography import REAL_GEOGRAPHY
                        city_name_map = {c["id"]: c["name"] for c in REAL_GEOGRAPHY["customers"]}
                        skipped_name = city_name_map.get(skipped_node, f"Node {skipped_node}")
                    st.markdown(f"**📍 Customer Being Skipped: {skipped_name}**")


def create_quality_figure_for_vehicle(inst, sim_res, vehicle_k):
    """Quality degradation over ACTUAL simulation time (0→300 min) using log_rows"""
    state = sim_res.final_states.get(vehicle_k)
    if not state or not state.batch_metrics:
        fig = go.Figure()
        fig.add_annotation(text="No batch data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(height=550, title=f"Vehicle {vehicle_k}")
        return fig

    # Build per-batch time-series from log_rows
    batch_timeseries = {}   # batch_id -> {"t": [], "q": []}
    for row in sim_res.log_rows:
        if row.get("vehicle") != vehicle_k:
            continue
        t = row["t_min"]
        for bid, q in row.get("batch_qualities", {}).items():
            if bid not in batch_timeseries:
                batch_timeseries[bid] = {"t": [], "q": []}
            batch_timeseries[bid]["t"].append(t)
            batch_timeseries[bid]["q"].append(q)

    # Build rich batch info: batch_id -> {produce, city, delivered}
    from real_geography import REAL_GEOGRAPHY
    city_name_map = {c["id"]: c["name"] for c in REAL_GEOGRAPHY["customers"]}

    # Which customer nodes were actually delivered this simulation?
    delivered_nodes = {
        ev.details.get("node")
        for ev in sim_res.events
        if ev.event == "SERVICE_START"
    }

    # batch_id -> (produce_type, city_name, delivered)
    batch_info = {}
    for s in inst.shipments:
        bid = s.batch.batch_id
        city = city_name_map.get(s.customer_node_id, f"C{s.customer_node_id}")
        delivered = s.customer_node_id in delivered_nodes
        batch_info[bid] = (s.batch.produce_type.capitalize(), city, delivered)

    colors = px.colors.qualitative.Bold
    fig = go.Figure()

    for idx, (batch_id, series) in enumerate(sorted(batch_timeseries.items())):
        produce, city, delivered = batch_info.get(
            batch_id, (f"Batch {batch_id}", "Unknown", True)
        )
        not_delivered_tag = " ❌ Not Delivered" if not delivered else ""
        label = f"{produce} · {city}{not_delivered_tag}"

        fig.add_trace(go.Scatter(
            x=series["t"], y=series["q"],
            mode="lines",
            name=label,
            line=dict(
                color=colors[idx % len(colors)],
                width=2.5,
                dash="dot" if not delivered else "solid"   # dashed if not delivered
            ),
            hovertemplate=f"<b>{label}</b><br>Sim Time: %{{x:.0f}} min<br>Quality: %{{y:.1%}}<extra></extra>"
        ))

    # Min quality threshold line
    fig.add_hline(y=0.60, line_dash="dash", line_color="#eb3349", line_width=2,
                  annotation_text="Min Quality (60%)",
                  annotation_position="bottom right",
                  annotation_font_color="#eb3349")

    # Mark reroute events as vertical lines
    for ev in sim_res.events:
        if ev.vehicle_id == vehicle_k and ev.event == "REROUTE_APPLIED":
            fig.add_vline(x=ev.t_min, line_dash="dot", line_color="#f2994a", line_width=2,
                          annotation_text=f"🔄 Reroute",
                          annotation_font_color="#f2994a", annotation_position="top left")

    fig.update_layout(
        height=550,
        showlegend=True,
        title=dict(text=f"🚚 Vehicle {vehicle_k} — Quality Degradation Over Travel Time",
                   font=dict(size=15, color="#667eea"), x=0.5, xanchor="center"),
        xaxis_title="Travel Time (min)",
        yaxis_title="Quality Remaining",
        yaxis=dict(range=[0, 1.08], tickformat=".0%", gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(
            orientation="v",
            x=0.01, y=0.01,          # inside bottom-left — always visible
            xanchor="left", yanchor="bottom",
            font=dict(size=12, color="white"),
            bgcolor="rgba(20,20,35,0.75)",
            bordercolor="#667eea", borderwidth=1
        ),
        margin=dict(l=50, r=40, t=60, b=50),   # reduced right margin (legend is inside now)
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.6)",
        font=dict(color="white")
    )
    return fig


def create_temperature_figure_for_vehicle(inst, sim_res, vehicle_k):
    """Per-vehicle compartment temperature over ACTUAL travel time (0→300 min) using log_rows"""
    state = sim_res.final_states.get(vehicle_k)
    vehicle_meta = inst.vehicle_meta.get(vehicle_k)
    if not state or not vehicle_meta:
        fig = go.Figure()
        fig.add_annotation(text="No temperature data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(height=480, title=f"Vehicle {vehicle_k}")
        return fig

    # Build per-compartment time-series from log_rows
    # log_rows has: {t_min, vehicle, comp_temps: {comp_id: temp}}
    comp_timeseries = {}   # comp_id -> {"t": [], "temp": []}
    for row in sim_res.log_rows:
        if row.get("vehicle") != vehicle_k:
            continue
        t = row["t_min"]
        for cid, temp in row.get("comp_temps", {}).items():
            if cid not in comp_timeseries:
                comp_timeseries[cid] = {"t": [], "temp": []}
            comp_timeseries[cid]["t"].append(t)
            comp_timeseries[cid]["temp"].append(temp)

    comp_label_map = {"A": "Compartment A: Dairy", "B": "Compartment B: Produce", "C": "Compartment C: Flowers"}
    comp_color_map = {"A": "#4169E1", "B": "#56ab2f", "C": "#f2994a"}

    fig = go.Figure()

    for comp_id, series in sorted(comp_timeseries.items()):
        label = comp_label_map.get(comp_id, f"Compartment {comp_id}")
        color = comp_color_map.get(comp_id, "#ffffff")
        setpoint = vehicle_meta.compartments[comp_id].setpoint_c

        # Actual Temp Line
        fig.add_trace(go.Scatter(
            x=series["t"], y=series["temp"],
            mode="lines",
            name=f"{label}",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{label}</b><br>Travel Time: %{{x:.0f}} min<br>Temperature: %{{y:.2f}}°C<extra></extra>"
        ))
        
        # Setpoint Line (Dashed)
        fig.add_trace(go.Scatter(
            x=[series["t"][0], series["t"][-1]] if series["t"] else [0, 1],
            y=[setpoint, setpoint],
            mode="lines",
            name=f"{label} (Setpoint)",
            line=dict(color=color, width=1.5, dash="dash"),
            opacity=0.6,
            hovertemplate=f"Setpoint: %{{y:.1f}}°C<extra></extra>",
            showlegend=False  # Hide setpoint from legend to reduce clutter, it's obvious from color
        ))

    # Mark reroute events as vertical lines
    for ev in sim_res.events:
        if ev.vehicle_id == vehicle_k and ev.event == "REROUTE_APPLIED":
            fig.add_vline(x=ev.t_min, line_dash="dot", line_color="#f2994a", line_width=2,
                          annotation_text=f"🔄 Reroute",
                          annotation_font_color="#f2994a", annotation_position="top left")

    fig.update_layout(
        height=480,
        showlegend=True,
        title=dict(text=f"🌡️ Vehicle {vehicle_k} — Compartment Temperatures Over Time",
                   font=dict(size=15, color="#667eea"), x=0.5, xanchor="center"),
        xaxis_title="Travel Time (min)",
        yaxis_title="Temperature (°C)",
        legend=dict(
            orientation="v",
            x=0.01, y=0.01,          # inside bottom-left
            xanchor="left", yanchor="bottom",
            font=dict(size=12, color="white"),
            bgcolor="rgba(20,20,35,0.75)",
            bordercolor="#667eea", borderwidth=1
        ),
        margin=dict(l=50, r=40, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.6)",
        font=dict(color="white"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


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
        
        # ── Live Route Map Section ─────────────────────────────────────────────
        st.markdown("### 🗺️ Live Route Visualization")

        max_sim_time = max(s.clock_min for s in sim_res.final_states.values()) if sim_res.final_states else 0

        speed_multiplier = st.radio(
            "▶ Playback Controls", 
            options=[0, 1, 5, 10, 30, 60], 
            format_func=lambda x: "⏸ Paused" if x == 0 else ("1x (Real Time)" if x == 1 else f"{x}x Fast Forward"),
            horizontal=True
        )

        current_real_time = time.time()
        
        # Initialize playback state
        if 'last_real_time' not in st.session_state:
            st.session_state['last_real_time'] = current_real_time
        if 'current_sim_time' not in st.session_state:
            st.session_state['current_sim_time'] = 0.0

        # Calculate time delta since last frame
        delta_real_seconds = current_real_time - st.session_state['last_real_time']
        st.session_state['last_real_time'] = current_real_time

        # If playing, auto-advance the internal time
        if speed_multiplier > 0:
            sim_minutes_delta = (delta_real_seconds / 60.0) * speed_multiplier
            st.session_state['current_sim_time'] = min(st.session_state['current_sim_time'] + sim_minutes_delta, max_sim_time)

        # Allow user to drag the slider manually
        current_slider_time = st.slider(
            "⏱ Simulation Time (minutes)",
            min_value=0.0,
            max_value=float(max_sim_time),
            value=float(st.session_state['current_sim_time']),
            step=1.0,
            format="%.1f min"
        )
        
        # Sync the session state with the slider's value
        # If the user drags it, it will overwrite the auto-advanced value
        if current_slider_time != float(st.session_state['current_sim_time']):
            st.session_state['current_sim_time'] = current_slider_time

        # Sleep briefly if playing to trigger a fast re-run loop
        if speed_multiplier > 0 and st.session_state['current_sim_time'] < max_sim_time:
            time.sleep(0.1)
            st.rerun()

        # Render Map at current time
        map_fig = create_live_folium_map(inst, sim_res, st.session_state['current_sim_time'])
        st_folium(map_fig, width=1200, height=500, returned_objects=[])

        st.markdown("---")

        st.markdown("### 🔍 Select Vehicle for Statistics")
        veh_keys = [None] + sorted(sim_res.final_states.keys())
        selected_vehicle = st.selectbox(
            "Choose a vehicle to view its specific metrics below:",
            options=veh_keys,
            format_func=lambda x: "Select a vehicle..." if x is None else f"Vehicle {x+1}"
        )

        st.markdown("---")

        if selected_vehicle is not None:
            # Reroute Timeline
            st.markdown(f"### 🔄 Rerouting Decisions (Vehicle {selected_vehicle+1})")
            display_reroute_timeline(sim_res, inst, res, selected_vehicle)
    
            st.markdown("---")
    
            # Vehicle Summary
            st.markdown(f"### 📊 Vehicle {selected_vehicle+1} Summary Statistics")
            k = selected_vehicle
            state = sim_res.final_states[k]
            route = state.route
            vehicle_dist = sum(
                inst.dist.get((route[i], route[i+1]), 0.0)
                for i in range(len(route) - 1)
            )
            vehicle_time = state.clock_min
            
            has_actual_reroute = any(ev.event == "REROUTE_APPLIED" and ev.vehicle_id == k for ev in sim_res.events)
            rerouted_label = "✅ Yes" if has_actual_reroute else "❌ No"
            route_str = " → ".join(
                "Depot" if n in (inst.start, inst.end) else (inst.node_meta[n].name if hasattr(inst, 'node_meta') and n in inst.node_meta else str(n))
                for n in route
            )
            
            card_html = f"""
            <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                        border:1px solid #667eea; border-radius:12px;
                        padding:1rem; margin-bottom:0.5rem;'>
                <h4 style='color:#667eea;margin:0'>🚚 Vehicle {k+1}</h4>
                <hr style='border-color:#667eea33;margin:0.5rem 0'/>
                <p style='margin:0.2rem 0'><b>Distance:</b> {vehicle_dist:.1f} km</p>
                <p style='margin:0.2rem 0'><b>Duration:</b> {vehicle_time:.0f} min</p>
                <p style='margin:0.2rem 0'><b>Rerouted:</b> {rerouted_label}</p>
                <hr style='border-color:#667eea33;margin:0.5rem 0'/>
                <p style='margin:0.2rem 0;font-size:0.72rem;line-height:1.4'><b>Route:</b> {route_str}</p>
            """
    
            # Find skipped customers for this vehicle
            skipped_names = []
            for ev in sim_res.events:
                if ev.event == "REROUTE_APPLIED" and ev.vehicle_id == k and ev.details.get('option_selected') == "skip_customer":
                    opt = ev.details.get('option_name', '')
                    try:
                        nid = int(opt.split("Skip customer ")[1].split(" ")[0])
                        skipped_names.append(inst.node_meta[nid].name if nid in inst.node_meta else str(nid))
                    except: pass
            
            # Find refused customers
            refused_names = []
            orig_route = res.routes.get(k, []) if hasattr(res, 'routes') else []
            for nid in orig_route:
                if nid in inst.customers:
                    served = any(e.event == "SERVICE_START" and e.vehicle_id == k and e.details.get('node') == nid for e in sim_res.events)
                    if not served:
                        cname = inst.node_meta[nid].name if nid in inst.node_meta else str(nid)
                        refused = any(e.event == 'SERVICE_REFUSED' and e.vehicle_id == k and e.details.get('node') == nid for e in sim_res.events)
                        if refused:
                            refused_names.append(cname)
                        elif cname not in skipped_names:
                            refused_names.append(cname)
    
            if skipped_names:
                card_html += f"<p style='margin:0.2rem 0;color:#ff9f43;font-size:0.7rem'><b>📉 Skipped:</b> {', '.join(skipped_names)}</p>"
            else:
                card_html += f"<p style='margin:0.2rem 0;color:#555;font-size:0.7rem'><b>📉 Skipped:</b> None</p>"
                
            if refused_names:
                card_html += f"<p style='margin:0.2rem 0;color:#ff6b81;font-size:0.7rem'><b>⚠️ Refused (Quality):</b> {', '.join(refused_names)}</p>"
            else:
                card_html += f"<p style='margin:0.2rem 0;color:#555;font-size:0.7rem'><b>⚠️ Refused (Quality):</b> None</p>"
                
            card_html += "</div>"
            
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("**Average temperature values:**")
            for comp, temp in state.compartment_temps.items():
                setpoint = inst.vehicle_meta[k].compartments[comp].setpoint_c
                dev = temp - setpoint
                icon = "🔴" if abs(dev) > 3 else "🟡" if abs(dev) > 1 else "🟢"
                st.write(f"{icon} **Compartment {comp}**: {temp:.2f}°C &nbsp; (setpoint {setpoint}°C, Δ{dev:+.2f}°C)")
    
            st.markdown("---")
    
            # Quality Degradation
            st.markdown(f"### 📈 Quality Degradation (Vehicle {selected_vehicle+1})")
            q_fig = create_quality_figure_for_vehicle(inst, sim_res, selected_vehicle)
            st.plotly_chart(q_fig, use_container_width=True)
    
            st.markdown("---")
    
            # Temperature Status
            st.markdown(f"### 🌡️ Temperature Status (Vehicle {selected_vehicle+1})")
            t_fig = create_temperature_figure_for_vehicle(inst, sim_res, selected_vehicle)
            st.plotly_chart(t_fig, use_container_width=True)
    
            st.markdown("---")

        # Customer Fulfillment Table logic MUST run for the PDF regardless of selection
        from real_geography import REAL_GEOGRAPHY
        city_name_map = {c["id"]: c["name"] for c in REAL_GEOGRAPHY["customers"]}

        # Determine which vehicle each customer was originally assigned to from res.routes
        customer_to_veh = {}
        if res and hasattr(res, 'routes'):
            for k, route in res.routes.items():
                for node in route:
                    if node in inst.customers:
                        customer_to_veh[node] = k

        customer_data = []
        for customer in inst.customers:
            served = any(ev.event == "SERVICE_START" and ev.details.get('node') == customer
                         for ev in sim_res.events)
            shipment = next((s for s in inst.shipments if s.customer_node_id == customer), None)
            if shipment:
                batch = shipment.batch
                vehicle_k = customer_to_veh.get(customer, "Unknown")
                city_name = city_name_map.get(customer, f"C{customer}")

                if served:
                    for k, state in sim_res.final_states.items():
                        if batch.batch_id in state.batch_metrics:
                            quality = estimate_quality_remaining(batch, state.batch_metrics[batch.batch_id])
                            vehicle_k = k  # update to actual vehicle that served it if it changed
                            break
                    else:
                        quality = 0
                else:
                    quality = None
                    
                # Determine failure reason if not delivered
                reason = "-"
                if not served:
                    refused = any(ev.event == "SERVICE_REFUSED" and ev.details.get('node') == customer
                                  for ev in sim_res.events)
                    if refused:
                        reason = "⚠️ Quality Below Threshold"
                    elif vehicle_k != "Unknown":
                        reason = "📉 Skipped (Autonomous Reroute)"
                    else:
                        reason = "❌ Unassigned"

                customer_data.append({
                    'Customer': city_name,
                    'Product': batch.produce_type.capitalize(),
                    'Units': shipment.demand_units,
                    'Vehicle': f"Vehicle {vehicle_k+1}" if vehicle_k != "Unknown" else "Unassigned",
                    'Status': '✅ Delivered' if served else '❌ Not Delivered',
                    'Final Quality Delivered': f'{quality:.1%}' if quality is not None else 'N/A',
                    'Failure Reason': reason
                })
        df_customers = pd.DataFrame(customer_data)

        if selected_vehicle is not None:
            st.markdown("### 👥 Customer Fulfillment Details")
            st.dataframe(df_customers[df_customers['Vehicle'] == f"Vehicle {selected_vehicle+1}"], use_container_width=True, hide_index=True)

        # ── Download Section ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📦 Download Simulation Results")
        st.markdown("<p style='color:#aaa;font-size:0.85rem'>All files are generated from the latest simulation run.</p>", unsafe_allow_html=True)

        import io, json, zipfile
        from datetime import datetime
        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # ── 1. PDF Report ─────────────────────────────────────────────────────────
        pdf_bytes = generate_pdf_report(inst, sim_res, res, df_customers, fulfillment_rate, avg_quality, reroute_count)

        st.markdown(f"""
        <div style='border:1px solid #667eea44; border-radius:10px; padding:1.5rem;
                    background:linear-gradient(135deg,#1a1a2e,#16213e);
                    text-align:center; margin-bottom:1rem'>
            <div style='font-size:2rem; margin-bottom:0.5rem;'>📄</div>
            <div style='font-size:1.4rem'>Comprehensive Simulation Report</div>
            <div style='font-size:0.9rem;color:#aaa;margin-bottom:1rem'>Contains fulfillment rates, vehicle summaries, routes, and rerouting logs.</div>
        </div>""", unsafe_allow_html=True)
        
        st.download_button(
            label="⬇ Download PDF Report", 
            data=pdf_bytes, 
            file_name=f"simulation_report_{run_ts}.pdf", 
            mime="application/pdf", 
            use_container_width=True, 
            key="dl_pdf_only"
        )

    
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

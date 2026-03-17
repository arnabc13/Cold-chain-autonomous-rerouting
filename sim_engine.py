# sim_engine.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from config import SimConfig
from data_models import PlanningInstance, VehicleSimState, ExposureMetrics
from temperature_model import update_compartment_temperature
from monitoring import update_exposure_metrics, estimate_quality_remaining
from reroute_engine import (
    generate_reroute_options,
    select_best_reroute_option,
    apply_reroute
)


@dataclass
class SimEvent:
    t_min: float
    vehicle_id: int
    event: str
    details: Dict[str, Any]


@dataclass
class SimulationResult:
    log_rows: List[Dict[str, Any]]          # time-step log (can be DF later)
    events: List[SimEvent]                 # discrete events
    final_states: Dict[int, VehicleSimState]


def _arc_distance(inst: PlanningInstance, i: int, j: int) -> float:
    return inst.dist[(i, j)]


def _arc_travel_time(inst: PlanningInstance, i: int, j: int) -> float:
    return inst.tt[(i, j)]


def simulate_routes(
    inst: PlanningInstance,
    routes: Dict[int, List[int]],
    cfg: SimConfig,
    seed: int = 7,
    enable_reroute_triggers: bool = True,
) -> SimulationResult:
    """
    Time-stepped simulation (NOT real-time):
      - updates vehicle movement along the planned route
      - generates temperature time-series via temperature_model
      - updates monitoring metrics via monitoring module
      - triggers reroute "recommendation events" (no auto-apply yet)

    Key outputs:
      - per-step log rows
      - event list
      - final per-vehicle state
    """
    random.seed(seed)

    dt = float(cfg.dt_min)
    horizon = float(cfg.horizon_min)

    # Pre-index shipments by customer node for quick lookup
    ship_by_node = {sh.customer_node_id: sh for sh in inst.shipments}

    # Initialize vehicle states
    states: Dict[int, VehicleSimState] = {}
    for k, route in routes.items():
        if not route or len(route) < 2:
            continue

        # start at depot
        s = route[0]
        nxt = route[1]
        
        # Initialize compartment temperatures from vehicle configuration
        vehicle = inst.vehicle_meta[k]
        comp_temps = {comp_id: comp.temp_c for comp_id, comp in vehicle.compartments.items()}
        
        states[k] = VehicleSimState(
            vehicle_id=k,
            route=route,
            route_index=0,
            remaining_dist_to_next=_arc_distance(inst, s, nxt),
            current_node=s,
            next_node=nxt,
            clock_min=0.0,
            compartment_temps=comp_temps,  # Per-compartment temps
            batch_metrics={},  # Per-batch metrics (populated during simulation)
            delayed_min=0.0,
            reroute_triggered=False,
        )

    log_rows: List[Dict[str, Any]] = []
    events: List[SimEvent] = []

    # Simulation loop
    sim_t = 0.0
    while sim_t <= horizon:
        for k, st in states.items():
            veh = inst.vehicle_meta[k]

            # if already finished (at end depot), keep logging minimal / skip
            if st.next_node is None:
                continue

            # Determine whether we are "at a stop" this step (door open)
            door_open = False

            # Random delay events (road disruptions)
            if random.random() < cfg.delay_event_prob:
                st.delayed_min += cfg.delay_event_min
                events.append(
                    SimEvent(sim_t, k, "DELAY_EVENT", {"delay_added_min": cfg.delay_event_min})
                )

            # Movement progress this step:
            # distance progress = (dt / travel_time) * arc_distance
            # add noise + speed factor
            i, j = st.current_node, st.next_node
            arc_dist = _arc_distance(inst, i, j)
            arc_tt = _arc_travel_time(inst, i, j)

            # progress fraction for this step
            # if arc_tt is big, fraction is small
            base_frac = dt / max(1e-6, arc_tt)

            # speed factor + road noise
            frac = base_frac * veh.max_speed_factor * max(0.2, random.gauss(1.0, cfg.travel_time_noise_sd))

            # If delayed, consume delay first (no movement)
            if st.delayed_min > 0:
                consume = min(st.delayed_min, dt)
                st.delayed_min -= consume
                frac = 0.0  # no movement this step if delay is active

            dist_progress = frac * arc_dist
            st.remaining_dist_to_next = max(0.0, st.remaining_dist_to_next - dist_progress)

            arrived = st.remaining_dist_to_next <= 1e-6

            # If arrived at next node, apply service time as "waiting"
            if arrived:
                st.route_index += 1
                st.current_node = j

                # door open at customer stops (not at depot)
                if j in inst.customers:
                    door_open = True
                    
                    # PRE-DELIVERY QUALITY CHECK: Refuse service if quality below threshold
                    if j in ship_by_node:
                        shipment = ship_by_node[j]
                        batch_id = shipment.batch.batch_id
                        if batch_id in st.batch_metrics:
                            batch = shipment.batch
                            quality = estimate_quality_remaining(batch, st.batch_metrics[batch_id])
                            
                            if quality < cfg.trigger_min_quality:
                                print(f"  [REFUSED] t={sim_t:.1f} veh={k} customer={j} quality={quality:.1%} < {cfg.trigger_min_quality:.1%}")
                                events.append(SimEvent(sim_t, k, "SERVICE_REFUSED", {"node": j, "quality": round(quality, 3)}))
                                # Move to next node without service
                                if st.route_index >= len(st.route) - 1:
                                    st.next_node = None
                                else:
                                    st.next_node = st.route[st.route_index + 1]
                                    st.remaining_dist_to_next = _arc_distance(inst, st.current_node, st.next_node)
                                continue

                # Apply service time as extra local time “spent” (we model it as added delay)
                service_time = inst.service[j]
                if service_time > 0:
                    st.delayed_min += service_time
                    events.append(
                        SimEvent(sim_t, k, "SERVICE_START", {"node": j, "service_min": service_time})
                    )

                # Advance to next node if any
                if st.route_index >= len(st.route) - 1:
                    # reached end depot
                    st.next_node = None
                    events.append(
                        SimEvent(sim_t, k, "ROUTE_COMPLETE", {"end_node": j})
                    )
                else:
                    st.next_node = st.route[st.route_index + 1]
                    st.remaining_dist_to_next = _arc_distance(inst, st.current_node, st.next_node)

            # --- Temperature update per compartment ---
            ambient = cfg.ambient_temp_c
            vehicle = inst.vehicle_meta[k]
            
            # Determine which compartment(s) have door open (at customer stops)
            doors_open = set()
            if arrived and j in inst.customers:
                # Find which compartment this customer's shipment is in
                if j in ship_by_node:
                    customer_comp = ship_by_node[j].assigned_compartment
                    doors_open.add(customer_comp)
            
            # Update each compartment's temperature independently
            for comp_id, comp in vehicle.compartments.items():
                door_open_this_comp = comp_id in doors_open
                new_temp = update_compartment_temperature(
                    compartment=comp,
                    ambient_c=ambient,
                    cfg=cfg,
                    door_open=door_open_this_comp
                )
                # Update both vehicle metadata and state tracking
                comp.temp_c = new_temp
                st.compartment_temps[comp_id] = new_temp

            # --- Monitoring update: per-batch tracking ---
            # Group shipments by compartment on this vehicle's route
            from collections import defaultdict
            shipments_by_comp = defaultdict(list)
            remaining_nodes = set(st.route[st.route_index:])
            for n in remaining_nodes:
                if n in ship_by_node:
                    sh = ship_by_node[n]
                    shipments_by_comp[sh.assigned_compartment].append(sh)
            
            # Track each batch individually using its compartment's temperature
            batch_qualities = {}
            for comp_id, shipments in shipments_by_comp.items():
                comp_temp = st.compartment_temps.get(comp_id, 10.0)
                
                for shipment in shipments:
                    batch = shipment.batch
                    batch_id = batch.batch_id
                    
                    # Initialize metrics for this batch if needed
                    if batch_id not in st.batch_metrics:
                        st.batch_metrics[batch_id] = ExposureMetrics()
                    
                    # Update metrics using compartment temperature
                    st.batch_metrics[batch_id] = update_exposure_metrics(
                        st.batch_metrics[batch_id],
                        temp_c=comp_temp,
                        batch=batch,
                        dt_min=dt
                    )
                    
                    # Calculate quality for this batch
                    batch_qualities[batch_id] = estimate_quality_remaining(
                        batch, st.batch_metrics[batch_id]
                    )

            # --- Trigger logic (reroute recommendation events only) ---
            # Check for violations across all batches
            if enable_reroute_triggers and not st.reroute_triggered:
                trigger_reason = None
                worst_batch_id = None
                
                for batch_id, metrics in st.batch_metrics.items():
                    # 1) continuous excursion too long
                    if metrics.max_continuous_excursion_min >= cfg.trigger_excursion_min:
                        trigger_reason = "TEMP_EXCURSION"
                        worst_batch_id = batch_id
                        break
                    
                    # 2) cumulative abuse too high
                    if metrics.cumulative_abuse >= cfg.trigger_abuse_units:
                        trigger_reason = trigger_reason or "CUMULATIVE_ABUSE"
                        worst_batch_id = batch_id
                    
                    # 3) NEW: Quality dropped below minimum acceptable threshold
                    batch = next((s.batch for s in inst.shipments if s.batch.batch_id == batch_id), None)
                    if batch:
                        quality = estimate_quality_remaining(batch, metrics)
                        # DEBUG: Print quality checks
                        if quality < 0.70:  # Show when quality is getting low
                            print(f"  [DEBUG] t={sim_t:.1f} veh={k} batch={batch_id} quality={quality:.1%} threshold={cfg.trigger_min_quality:.1%}")
                        if quality < cfg.trigger_min_quality:
                            print(f"  [TRIGGER] Quality trigger fired! batch={batch_id} quality={quality:.1%}")
                            trigger_reason = trigger_reason or "LOW_QUALITY"
                            worst_batch_id = batch_id

                # 3) late risk: compare current clock to TW close of next customer (rough)
                if st.next_node is not None and st.next_node in inst.customers:
                    tw_open, tw_close = inst.tw[st.next_node]
                    # crude ETA: assume remaining arc travel time proportional to remaining distance
                    eta = sim_t + (_arc_travel_time(inst, st.current_node, st.next_node) *
                                   (st.remaining_dist_to_next / max(1e-6, _arc_distance(inst, st.current_node, st.next_node))))
                    if eta > tw_close + cfg.trigger_late_by_min:
                        trigger_reason = trigger_reason or "PREDICTED_LATE"
                        if worst_batch_id is None and st.batch_metrics:
                            worst_batch_id = list(st.batch_metrics.keys())[0]

                if trigger_reason is not None:
                    # Generate reroute options
                    options = generate_reroute_options(
                        inst, st, ship_by_node, worst_batch_id, sim_t, cfg
                    )
                    
                    if options and cfg.enable_auto_reroute:
                        # Find violated shipment for scoring
                        violated_shipment = next(
                            (s for s in inst.shipments if s.batch.batch_id == worst_batch_id),
                            None
                        )
                        
                        if violated_shipment:
                            # Auto-select best option
                            best_option = select_best_reroute_option(
                                options, violated_shipment, cfg
                            )
                            
                            # Apply new route immediately
                            apply_reroute(st, best_option.new_route, st.current_node)
                            
                            # Update distance to next node after reroute
                            # Guard: skip if next_node == current_node (self-loop) to avoid KeyError
                            if st.next_node is not None and st.next_node != st.current_node:
                                st.remaining_dist_to_next = _arc_distance(
                                    inst, st.current_node, st.next_node
                                )
                            elif st.next_node == st.current_node:
                                # Advance one more step to a truly different node
                                _idx = st.route_index + 2
                                st.next_node = st.route[_idx] if _idx < len(st.route) else None
                                if st.next_node is not None and st.next_node != st.current_node:
                                    st.remaining_dist_to_next = _arc_distance(inst, st.current_node, st.next_node)
                            
                            # Mark as rerouted
                            st.reroute_triggered = True
                            
                            # Log autonomous reroute decision
                            events.append(
                                SimEvent(sim_t, k, "REROUTE_APPLIED", {
                                    "reason": trigger_reason,
                                    "batch_id": worst_batch_id,
                                    "option_selected": best_option.option_type,
                                    "option_name": best_option.name,
                                    "customers_lost": best_option.customers_lost,
                                    "expected_quality_gain": round(best_option.expected_quality_gain, 3),
                                    "score": round(best_option.estimated_score, 2),
                                    "new_route": best_option.new_route,
                                    "alternatives_considered": len(options),
                                    "current_node": st.current_node,
                                })
                            )
                        else:
                            # Fallback if shipment not found
                            st.reroute_triggered = True
                            worst_metrics = st.batch_metrics.get(worst_batch_id, ExposureMetrics())
                            worst_quality = batch_qualities.get(worst_batch_id, None)
                            events.append(
                                SimEvent(sim_t, k, "REROUTE_RECOMMENDATION", {
                                    "reason": trigger_reason,
                                    "batch_id": worst_batch_id,
                                    "above_safe_min": round(worst_metrics.above_safe_minutes, 1),
                                    "cumulative_abuse": round(worst_metrics.cumulative_abuse, 1),
                                    "quality_remaining": None if worst_quality is None else round(worst_quality, 3),
                                })
                            )
                    else:
                        # Auto-reroute disabled or no options available
                        st.reroute_triggered = True
                        worst_metrics = st.batch_metrics.get(worst_batch_id, ExposureMetrics())
                        worst_quality = batch_qualities.get(worst_batch_id, None)
                        events.append(
                            SimEvent(sim_t, k, "REROUTE_RECOMMENDATION", {
                                "reason": trigger_reason,
                                "batch_id": worst_batch_id,
                                "above_safe_min": round(worst_metrics.above_safe_minutes, 1),
                                "cumulative_abuse": round(worst_metrics.cumulative_abuse, 1),
                                "quality_remaining": None if worst_quality is None else round(worst_quality, 3),
                                "current_node": st.current_node,
                                "next_node": st.next_node,
                                "remaining_stops": list(st.route[st.route_index+1:]),
                            })
                        )

            # Log row - now with compartment-level data
            # Average temperatures and quality across compartments
            avg_temp = sum(st.compartment_temps.values()) / len(st.compartment_temps) if st.compartment_temps else 10.0
            avg_quality = sum(batch_qualities.values()) / len(batch_qualities) if batch_qualities else None
            total_above_safe = sum(m.above_safe_minutes for m in st.batch_metrics.values())
            total_above_critical = sum(m.above_critical_minutes for m in st.batch_metrics.values())
            max_excursion = max((m.max_continuous_excursion_min for m in st.batch_metrics.values()), default=0.0)
            total_abuse = sum(m.cumulative_abuse for m in st.batch_metrics.values())
            
            log_rows.append({
                "t_min": sim_t,
                "vehicle": k,
                "current_node": st.current_node,
                "next_node": st.next_node,
                "remaining_dist_to_next": round(st.remaining_dist_to_next, 3) if st.next_node is not None else 0.0,
                "avg_temp_c": round(avg_temp, 2),
                "comp_temps": {comp_id: round(temp, 2) for comp_id, temp in st.compartment_temps.items()},
                "total_above_safe_min": round(total_above_safe, 1),
                "total_above_critical_min": round(total_above_critical, 1),
                "max_excursion_min": round(max_excursion, 1),
                "total_cumulative_abuse": round(total_abuse, 2),
                "avg_quality_remaining": None if avg_quality is None else round(avg_quality, 3),
                "num_batches_tracked": len(st.batch_metrics),
                "delayed_min": round(st.delayed_min, 1),
                "reroute_triggered": st.reroute_triggered,
                # Per-batch quality for graphing
                "batch_qualities": {bid: round(q, 3) for bid, q in batch_qualities.items()},
            })

            # Advance state time (vehicle clock not strictly needed in this v1)
            st.clock_min = sim_t

        sim_t += dt

    return SimulationResult(log_rows=log_rows, events=events, final_states=states)

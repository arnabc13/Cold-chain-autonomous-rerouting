# reroute_engine.py
"""
Autonomous rerouting engine for cold chain optimization.

Handles:
- Detection of reroute triggers
- Generation of alternative route options
- Scoring and selection of best option
- Application of new routes during simulation
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import SimConfig
from data_models import PlanningInstance, VehicleSimState, Shipment


@dataclass
class RerouteOption:
    """Represents a single reroute alternative."""
    option_type: str               # "skip_customer", "return_to_depot", "continue_original"
    name: str                      # Human-readable description
    new_route: List[int]           # Modified route (node sequence)
    customers_lost: int            # Number of customers not served
    expected_quality_gain: float   # Estimated quality improvement (0-1)
    additional_delay_min: float    # Extra time required
    distance_saved_km: float       # Distance saved (for early return)
    estimated_score: float = 0.0   # Calculated score (higher = better)


def generate_reroute_options(
    inst: PlanningInstance,
    vehicle_state: VehicleSimState,
    ship_by_node: Dict[int, Shipment],
    violated_batch_id: int,
    current_time: float,
    cfg: SimConfig
) -> List[RerouteOption]:
    """
    Generate alternative routes when a reroute trigger fires.
    
    Returns 3 options:
    1. Skip lowest-priority customer
    2. Return to depot immediately
    3. Continue with original route (baseline)
    """
    options = []
    
    # Get remaining route — start from route_index+1 to EXCLUDE the current node.
    # The trigger fires after service at the current node, so current_node has
    # already been served. Only truly future (unvisited) nodes are skip candidates.
    remaining_route = vehicle_state.route[vehicle_state.route_index + 1:]
    remaining_customers = [n for n in remaining_route if n in inst.customers]
    
    if not remaining_customers:
        # Already at end, no reroute needed
        return []
    
    # --- Option 1: Skip lowest-priority customer ---
    # Find customer with lowest priority (highest priority number)
    customer_priorities = {}
    for node in remaining_customers:
        if node in ship_by_node:
            customer_priorities[node] = ship_by_node[node].batch.priority
    
    if customer_priorities:
        # Skip customer with lowest priority (highest number)
        skip_customer = max(customer_priorities, key=customer_priorities.get)
        
        # Build new route without this customer
        new_route = [n for n in remaining_route if n != skip_customer]
        
        # Estimate time saved
        time_saved = inst.service.get(skip_customer, 5.0)
        if skip_customer in remaining_route[:-1]:  # Not last customer
            next_idx = remaining_route.index(skip_customer) + 1
            if next_idx < len(remaining_route):
                # Time saved = travel to skipped + service + travel from skipped
                # Approximation: avg of arcs involving this customer
                time_saved += 10.0  # Rough estimate
        
        # Quality gain: if we save time, temperature violations reduce
        quality_gain = min(0.15, time_saved / 60.0)  # Up to 15% improvement
        
        options.append(RerouteOption(
            option_type="skip_customer",
            name=f"Skip customer {skip_customer} (priority {customer_priorities[skip_customer]})",
            new_route=new_route,
            customers_lost=1,
            expected_quality_gain=quality_gain,
            additional_delay_min=0.0,
            distance_saved_km=0.0
        ))
    
    # --- Option 2: Return to depot immediately ---
    # **CONSTRAINT**: Only allow this option if abandoning ≤3 customers
    # (Don't want to abandon too many customers just for quality preservation)
    if len(remaining_customers) <= 3:
        # Abandon all remaining customers, go straight to end depot
        new_route = [vehicle_state.current_node, inst.end]
        
        # Calculate distance/time saved
        total_remaining_time = 0.0
        total_remaining_dist = 0.0
        for i in range(len(remaining_route) - 1):
            arc = (remaining_route[i], remaining_route[i+1])
            if arc in inst.tt:
                total_remaining_time += inst.tt[arc]
            if arc in inst.dist:
                total_remaining_dist += inst.dist[arc]
        
        # Time to go directly to depot
        direct_arc = (vehicle_state.current_node, inst.end)
        direct_time = inst.tt.get(direct_arc, 20.0)
        direct_dist = inst.dist.get(direct_arc, 20.0)
        
        time_saved = max(0.0, total_remaining_time - direct_time)
        dist_saved = max(0.0, total_remaining_dist - direct_dist)
        
        # Quality gain: significant time reduction
        quality_gain = min(0.30, time_saved / 120.0)  # Up to 30% improvement
        
        options.append(RerouteOption(
            option_type="return_to_depot",
            name=f"Return to depot immediately (abandon {len(remaining_customers)} customers)",
            new_route=new_route,
            customers_lost=len(remaining_customers),
            expected_quality_gain=quality_gain,
            additional_delay_min=-time_saved,  # Negative = faster
            distance_saved_km=dist_saved
        ))
    
    # --- Option 3: Continue with original route (baseline) ---
    options.append(RerouteOption(
        option_type="continue_original",
        name="Continue with original route (accept quality loss)",
        new_route=remaining_route,
        customers_lost=0,
        expected_quality_gain=0.0,  # No improvement
        additional_delay_min=0.0,
        distance_saved_km=0.0
    ))
    
    return options


def calculate_reroute_score(
    option: RerouteOption,
    violated_batch: Shipment,
    cfg: SimConfig
) -> float:
    """
    Score a reroute option based on cost/benefit analysis.
    
    Higher score = better option.
    
    Considers:
    - Revenue loss from skipped customers
    - Quality preservation (spoilage reduction)
    - Operational costs (time, fuel)
    - Policy preference (conservative vs aggressive)
    """
    score = 0.0
    
    # 1. Revenue impact (negative for lost customers)
    revenue_loss = option.customers_lost * cfg.revenue_per_customer
    score -= revenue_loss
    
    # 2. Quality preservation (positive for reduced spoilage)
    batch_value = violated_batch.demand_units * cfg.spoilage_cost_per_unit
    quality_benefit = option.expected_quality_gain * batch_value
    score += quality_benefit
    
    # 3. Time penalty (operational cost)
    if option.additional_delay_min > 0:
        score -= option.additional_delay_min * cfg.delay_cost_per_min
    else:
        # Time saved is a benefit
        score += abs(option.additional_delay_min) * cfg.delay_cost_per_min * 0.5
    
    # 4. Fuel savings (for early return)
    if option.option_type == "return_to_depot":
        score += option.distance_saved_km * cfg.fuel_cost_per_km
    
    # 5. Policy adjustments
    if cfg.reroute_policy == "conservative":
        # Prioritize not losing customers
        score -= option.customers_lost * 2000  # Extra penalty
    elif cfg.reroute_policy == "aggressive":
        # Prioritize quality preservation
        score += quality_benefit * 1.5  # Boost quality benefit
    # "balanced" uses base scoring
    
    return score


def select_best_reroute_option(
    options: List[RerouteOption],
    violated_batch: Shipment,
    cfg: SimConfig
) -> RerouteOption:
    """
    Score all options and select the best one.
    """
    if not options:
        raise ValueError("No reroute options available")
    
    # Score each option
    for option in options:
        option.estimated_score = calculate_reroute_score(option, violated_batch, cfg)
    
    # Select highest score
    best_option = max(options, key=lambda opt: opt.estimated_score)
    
    return best_option


def apply_reroute(
    vehicle_state: VehicleSimState,
    new_route: List[int],
    current_node: int
) -> None:
    """
    Apply a new route to the vehicle state.

    new_route contains ONLY future nodes (route_index+1 onward), without the
    skipped customer. We prepend the already-traveled portion to form the full route.

    Updates:
    - vehicle_state.route
    - vehicle_state.route_index
    - vehicle_state.next_node
    """
    # Everything up to and including current node (route_index) is already traveled
    traveled_path = vehicle_state.route[:vehicle_state.route_index + 1]

    # Safety: strip current_node from new_route in case it appears there
    # (would cause next_node == current_node → KeyError in _arc_distance)
    new_route = [n for n in new_route if n != current_node]

    # new_route = future nodes only (skipped customer absent, current_node absent)
    full_new_route = traveled_path + new_route

    # Update vehicle state
    vehicle_state.route = full_new_route
    vehicle_state.route_index = len(traveled_path) - 1  # Point to current node

    # Set next node
    if len(full_new_route) > vehicle_state.route_index + 1:
        vehicle_state.next_node = full_new_route[vehicle_state.route_index + 1]
    else:
        vehicle_state.next_node = None

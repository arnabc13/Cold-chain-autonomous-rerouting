# config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass(frozen=True)
class SimConfig:
    # --- Simulation clock ---
    dt_min: int = 5                  # time step in minutes
    horizon_min: int = 300           # total simulation horizon in minutes (demo)

    # --- Ambient environment (placeholder) ---
    ambient_temp_c: float = 25.0     # Cooler temperature (night delivery/mild weather)
    ambient_noise_sd: float = 0.4

    # --- Vehicle thermal behavior (placeholder physics) ---
    # cooling_power: how strongly vehicle pulls temp toward setpoint each step
    base_cooling_power: float = 0.3  # Stronger refrigeration system (was 0.25)
    # leakage: how strongly temp drifts toward ambient each step
    base_leakage: float = 0.06        # Reduced leakage (better insulation) (was 0.08)
    # door-open spike at customer stop (°C increase)
    door_open_spike_c: float = 1.8    # Reduced door opening impact (was 2.5)
    # probability of a reefer underperform step (temporary)
    reefer_glitch_prob: float = 0.16  # Fewer glitches (was 0.02)
    # if glitch, cooling power reduced by this factor
    glitch_cooling_factor: float = 0.50

    # --- Travel time variability (road conditions) ---
    # multiplicative noise on travel time progress
    travel_time_noise_sd: float = 0.12
    # probability of a random delay event per step
    delay_event_prob: float = 0.03
    # delay length in minutes when event happens
    delay_event_min: int = 10

    # --- Monitoring triggers (decision support) ---
    # trigger if continuous excursion above safe max exceeds this
    trigger_excursion_min: int = 60  # Wait for serious continuous violations
    # trigger if cumulative abuse exceeds this (arbitrary units)
    trigger_abuse_units: float = 150.0  # Allow more total abuse before triggering
    # trigger if predicted late (ETA > TW_close) by more than:
    trigger_late_by_min: int = 15
    # NEW: trigger if quality drops below this threshold (prevent spoiled deliveries)
    trigger_min_quality: float = 0.60  # Reroute if quality < 60% (prevent delivering bad products)

    # --- Autonomous rerouting configuration ---
    enable_auto_reroute: bool = True      # Enable automatic rerouting on triggers
    reroute_policy: str = "aggressive"      # "conservative", "aggressive", "balanced"
    
    # Cost parameters for reroute scoring (in currency units)
    revenue_per_customer: float = 1000.0      # Lost revenue if customer skipped (reduced to enable rerouting)
    spoilage_cost_per_unit: float = 2700.0    # High value: Quality-favored (skip wins often, but not always)
    delay_cost_per_min: float = 50.0          # Operational cost of delays
    fuel_cost_per_km: float = 10.0            # Fuel savings for early return
    reoptimization_cost: float = 500.0        # Computational overhead cost

    # --- Produce catalog placeholders ---
    # safe temp range (min,max), critical temp, shelf life hours, abuse sensitivity
    produce_catalog: Dict[str, Dict] = None
    
    # --- Multi-compartment configuration ---
    # Maps product types to appropriate compartment IDs
    product_to_compartment: Dict[str, str] = None
    # Defines compartment specifications
    compartment_specs: Dict[str, Dict] = None

    def __post_init__(self):
        # dataclass(frozen=True) blocks normal assignment, so we use object.__setattr__
        if self.produce_catalog is None:
            object.__setattr__(
                self,
                "produce_catalog",
                {
                    # TODO: replace with real commodity parameters
                    "tomato":   {"safe": (10.0, 15.0), "critical": 25.0, "shelf_life_h": 48.0, "k_abuse": 0.5},
                    "banana":   {"safe": (13.0, 16.0), "critical": 24.0, "shelf_life_h": 72.0, "k_abuse": 0.4},
                    "leafy":    {"safe": (2.0,  6.0),  "critical": 12.0, "shelf_life_h": 24.0, "k_abuse": 0.8},
                    "milk":     {"safe": (2.0,  4.0),  "critical": 8.0,  "shelf_life_h": 18.0, "k_abuse": 1.1},
                    "flowers":  {"safe": (1.0,  4.0),  "critical": 10.0, "shelf_life_h": 36.0, "k_abuse": 0.7},
                }
            )
        
        if self.product_to_compartment is None:
            object.__setattr__(
                self,
                "product_to_compartment",
                {
                    "milk":    "A",  # Cold compartment
                    "leafy":   "A",  # Cold compartment
                    "flowers": "A",  # Cold compartment
                    "tomato":  "B",  # Cool compartment
                    "banana":  "C",  # Ambient/warm compartment
                }
            )
        
        if self.compartment_specs is None:
            object.__setattr__(
                self,
                "compartment_specs",
                {
                    "A": {"setpoint_c": 3.0, "capacity_fraction": 0.4},   # 40% of vehicle
                    "B": {"setpoint_c": 12.0, "capacity_fraction": 0.35}, # 35% of vehicle
                    "C": {"setpoint_c": 15.0, "capacity_fraction": 0.25}, # 25% of vehicle
                }
            )

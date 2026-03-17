"""
Smoke tests for Cold Chain Autonomous Rerouting
These run on every git push via GitHub Actions CI.
If any test fails, the push is automatically reverted.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_imports():
    """Test that config module loads correctly"""
    from config import SimConfig
    cfg = SimConfig()
    assert cfg.dt_min == 5
    assert cfg.horizon_min == 300
    assert cfg.trigger_min_quality == 0.60
    print("✅ config.py imports OK")


def test_data_models_imports():
    """Test that data models load correctly"""
    from data_models import PlanningInstance
    print("✅ data_models.py imports OK")


def test_monitoring_imports():
    """Test that monitoring module loads correctly"""
    from monitoring import estimate_quality_remaining
    assert callable(estimate_quality_remaining)
    print("✅ monitoring.py imports OK")


def test_real_geography_imports():
    """Test that real geography data loads correctly"""
    from real_geography import REAL_GEOGRAPHY
    assert "hub" in REAL_GEOGRAPHY
    assert "customers" in REAL_GEOGRAPHY
    assert len(REAL_GEOGRAPHY["customers"]) == 12
    print("✅ real_geography.py — 12 customer cities verified OK")


def test_reroute_engine_imports():
    """Test that reroute engine loads correctly"""
    from reroute_engine import generate_reroute_options
    assert callable(generate_reroute_options)
    print("✅ reroute_engine.py imports OK")


def test_synthetic_data_imports():
    """Test that synthetic data generator loads correctly"""
    from synthetic_data import build_hub_to_city_instance
    assert callable(build_hub_to_city_instance)
    print("✅ synthetic_data.py imports OK")


def test_dashboard_syntax():
    """Test that dashboard.py has no syntax errors"""
    import ast
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard.py"
    )
    with open(dashboard_path, "r", encoding="utf-8") as f:
        source = f.read()
    # This will raise SyntaxError if dashboard.py has bad syntax
    ast.parse(source)
    print("✅ dashboard.py syntax check OK")


def test_requirements_file_exists():
    """Test that requirements.txt exists and has key packages"""
    req_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "requirements.txt"
    )
    assert os.path.exists(req_path), "requirements.txt missing!"
    with open(req_path) as f:
        content = f.read().lower()
    assert "streamlit" in content
    assert "plotly" in content
    assert "pandas" in content
    print("✅ requirements.txt exists with required packages")

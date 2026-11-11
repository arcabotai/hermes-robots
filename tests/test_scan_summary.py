"""LaserScan digest used by ros_echo (TurtleBot3 /scan: 360 beams, angle_min 0, + = left, null = out of range)."""
import math

from hermes_robots.backends.ros import scan_summary


def tb3_scan(ranges_by_deg, default=None):
    ranges = [default] * 360
    for deg, r in ranges_by_deg.items():
        ranges[deg % 360] = r
    return {"angle_min": 0.0, "angle_max": 6.28, "angle_increment": math.radians(1.0),
            "range_min": 0.12, "range_max": 3.5, "ranges": ranges}


def test_nearest_obstacle_and_sectors_follow_rep103():
    s = scan_summary(tb3_scan({0: 1.2, 90: 0.5, 270: 2.0, 180: 3.0}))
    assert s["nearest"] == {"distance_m": 0.5, "bearing_deg": 90.0}  # + = left
    sec = s["nearest_m_by_sector"]
    assert sec["front"] == 1.2 and sec["left"] == 0.5 and sec["right"] == 2.0 and sec["back"] == 3.0
    assert sec["front_left"] is None and sec["back_right"] is None
    assert s["beams"] == 360 and s["valid_returns"] == 4


def test_out_of_range_null_inf_nan_are_ignored():
    s = scan_summary(tb3_scan({0: 0.05, 10: float("inf"), 20: float("nan"), 30: 9.0, 40: "x", 350: 0.8}))
    assert s["valid_returns"] == 1
    assert s["nearest"] == {"distance_m": 0.8, "bearing_deg": -10.0}
    assert s["nearest_m_by_sector"]["front"] == 0.8


def test_open_space_has_no_nearest():
    s = scan_summary(tb3_scan({}))
    assert "nearest" not in s and all(v is None for v in s["nearest_m_by_sector"].values())


def test_non_scan_messages_are_untouched():
    assert scan_summary({"pose": {}}) is None
    assert scan_summary({"ranges": [1.0], "angle_min": 0.0}) is None  # no angle_increment
    assert scan_summary({"ranges": [1.0], "angle_min": 0.0, "angle_increment": 0.0}) is None
    assert scan_summary("nope") is None

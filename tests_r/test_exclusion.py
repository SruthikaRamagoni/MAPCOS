from mapcos.agents.exclusion_agent import ExclusionAgent

R = {"low": 1.0, "high": 5.0, "unit": "u", "source": "synthetic test range"}


def test_missing_values_unknown_not_normal():
    r = ExclusionAgent().run({})
    assert {c["status"] for c in r["checks"]} == {"UNKNOWN_MISSING_VALUE"} and not r["exclusion_screen_complete"]


def test_value_without_range_not_interpreted():
    r = ExclusionAgent().run({"tsh": {"value": 3.0}})
    assert r["checks"][0]["status"] == "UNKNOWN_NO_REFERENCE_RANGE"


def test_abnormal_high_and_low_flagged():
    r = ExclusionAgent().run({"tsh": {"value": 9.0, **R}, "prolactin": {"value": 0.2, **R}})
    assert [c["status"] for c in r["checks"]] == ["ABNORMAL_FLAG"] * 2
    assert r["is_confirmed_diagnosis"] is False and "NOT a confirmed diagnosis" in r["checks"][0]["message"]


def test_within_range_complete():
    r = ExclusionAgent().run({"tsh": {"value": 2.0, **R}, "prolactin": {"value": 3.0, **R}})
    assert r["exclusion_screen_complete"] and "unassessed" in r["summary"]


def test_configured_ranges_and_invalid_value():
    a = ExclusionAgent(reference_ranges={"tsh": R})
    assert a.run({"tsh": 2.0})["checks"][0]["status"] == "WITHIN_SUPPLIED_RANGE"
    assert a.run({"tsh": "abc"})["checks"][0]["status"] == "UNKNOWN_MISSING_VALUE"
    assert a.run(None)["unknown_tests"] == ["tsh", "prolactin"]


def test_not_assessed_listed():
    assert any("Cushing" in x for x in ExclusionAgent().run({})["not_assessed"])

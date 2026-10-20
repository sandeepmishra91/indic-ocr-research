from indic_ocr_research.text_audit import audit_pair, script_counts


def test_indic_combining_marks_are_counted_and_preserved():
    assert script_counts("का 12!") == {"Devanagari": 2}
    assert script_counts("മാ") == {"Malayalam": 2}
    audit = audit_pair("का", "ka", "hin")
    assert audit["reference_expected_script_fraction"] == 1
    assert audit["prediction_expected_script_fraction"] == 0
    assert not audit["empty_prediction"]


def test_empty_numeric_and_mixed_text():
    assert audit_pair("123", "", "mal")["reference_expected_script_fraction"] is None
    assert audit_pair("123", "", "mal")["empty_prediction"]
    mixed = audit_pair("कa", "कa", "hin")
    assert mixed["reference_expected_script_fraction"] == 0.5
    assert audit_pair("a", "a", "unknown")["reference_expected_script_fraction"] is None

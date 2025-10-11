import oscillink as ol


def test_exports_and_version_present():
    # Oscillink alias points to OscillinkLattice
    assert hasattr(ol, "Oscillink") and ol.Oscillink is ol.OscillinkLattice
    # version is a non-empty string
    assert isinstance(ol.__version__, str) and len(ol.__version__) > 0
    # a couple of public names in __all__
    assert set(["Oscillink", "verify_receipt"]).issubset(set(ol.__all__))

def test_colormaps():
    import matplotlib

    import cebra

    cmap = matplotlib.colormaps["cebra"]
    assert cmap is not None



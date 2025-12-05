from olmoearth_projects.utils.label_quality.spatial_extent import relative_area


def test_spatial_relative_extent() -> None:
    # Togo should be smaller than Nigeria, which should
    # be smaller than Mali
    togo_bb = (-0.0497847151599, 5.92883738853, 1.86524051271, 11.0186817489)
    nigeria_bb = (2.69170169436, 4.24059418377, 14.5771777686, 13.8659239771)
    mali_bb = (-12.1707502914, 10.0963607854, 4.27020999514, 24.9745740829)

    assert (
        0 < relative_area(togo_bb) < relative_area(nigeria_bb) < relative_area(mali_bb)
    )

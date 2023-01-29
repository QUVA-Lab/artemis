from artemis.general.geometry import reframe_from_a_to_b


def test_geometry():

    display_center_xy = 500, 500
    display_pointer_xy = 600, 300
    pixel_center_xy = 700, 700
    zoom = 2.
    pixel_xy = reframe_from_a_to_b(
        xy_in_a=display_pointer_xy,
        reference_xy_in_b=pixel_center_xy,
        reference_xy_in_a=display_center_xy,
        scale_in_a_of_b=zoom
    )

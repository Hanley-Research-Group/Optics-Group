from ctr_sim import Material


def test_material_creation():
    material = Material(
        name="Nitinol",
        youngs_modulus=60e9,
        shear_modulus=23e9,
    )

    assert material.name == "Nitinol"
    assert material.youngs_modulus == 60e9
    assert material.shear_modulus == 23e9
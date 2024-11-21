import pint


ureg = pint.UnitRegistry()


def to_quantity(value: float | pint.Quantity, unit: str) -> pint.Quantity:
    if isinstance(value, pint.Quantity):
        return value.to(unit)
    else:
        return value * ureg(unit)

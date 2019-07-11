from devito import DefaultDimension, Eq

__all__ = ['freesurface']


def freesurface(field, stencil_s, npml, forward=True):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation
    """
    fs = DefaultDimension(name="fs", default_value=stencil_s)

    field_m = field.forward if forward else field.backward

    lhs = field_m.subs({field.indices[-1]: npml - fs - 1})
    rhs = -field_m.subs({field.indices[-1]: npml + fs + 1})
    
    return [Eq(lhs, rhs), Eq(field_m.subs({field.indices[-1]: npml}), 0)]

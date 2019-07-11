from devito import DefaultDimension, Eq

__all__ = ['freesurface']


def freesurface(field, stencil_s, npml, forward=True):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation
    """
    fs = DefaultDimension(name="fs", default_value=stencil_s)

    t = field.indices[0] + (1 if forward else -1)
    grid = field.grid

    # lhs = field.subs({field.indices[-1]: npml - fs - 1})
    # rhs = -field.subs({field.indices[-1]: npml + fs + 1})
    
    if field.grid.dim ==1 :
        x = grid.dimensions[0]
        eqns = [Eq(field[t, npml - fs - 1], - field[t, npml + fs + 1])]
        eqns += [Eq(field[t, npml], 0)]
    elif field.grid.dim == 2:
        x, y = grid.dimensions
        eqns = [Eq(field[t, x, npml - fs - 1], -field[t, x, npml + fs + 1])]
        eqns += [Eq(field[t, x, npml], 0)]
    elif field.grid.dim == 3:
        x, y, z = grid.dimensions
        eqns = [Eq(field[t, x, y, npml - fs - 1], -field[t, x, y, npml + fs + 1])]
        eqns += [Eq(field[t, x, y, npml], 0)]
    else:
        return []
    return eqns

def effective_reference_fraction(override,configured,eligible_count):
    fraction=float(configured if override is None else override)
    if not 0<=fraction<=1:raise ValueError('Reference fraction must be within [0,1]')
    if fraction>0 and not eligible_count:raise ValueError('Positive reference fraction requires eligible reference examples')
    return fraction

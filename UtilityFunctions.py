"""
Includes useful functions for manipulating data
"""

from math import cos, sin

### Fast vector rotations for many points ###

def make_2d_rotation_transformation(angle, origin=[0, 0]):
    cos_theta, sin_theta = cos(angle), sin(angle)
    x0, y0 = origin
    def xform(point):
        x, y = point[0] - x0, point[1] - y0
        return [x * cos_theta - y * sin_theta + x0,
                x * sin_theta + y * cos_theta + y0]
    return xform

def fast_rotate(Points, angle, origin=[0, 0]):
    xform = make_2d_rotation_transformation(angle, origin)
    return [xform(v) for v in Points]


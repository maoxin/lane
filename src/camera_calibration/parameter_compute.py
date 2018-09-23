from sympy import Line, Point, intersection
import numpy as np

def vanishing_points(p1, p2, p3, p4):
    """
    road direction: p1, p2 | p3, p4
    perpendicular direction: p1, p4 | p2, p3
    """

    rl1 = Line(Point(p1), Point(p2))
    rl2 = Line(Point(p3), Point(p4))

    r_intersection = intersection(rl1, rl2)
    u0 = float(r_intersection[0][0])
    v0 = float(r_intersection[0][1])

    # pl1 = Line(Point(p1), Point(p4))
    pl1 = Line(Point(p2), Point(p3))
    hl = Line(Point(0, v0), (1, v0))

    p_intersection = intersection(pl1, hl)
    u1 = float(p_intersection[0][0])
    v1 = float(p_intersection[0][1])

    u0 = u0 - 1920 / 2
    u1 = u1 - 1920 / 2
    v0 = v0 - 1080 / 2

    return u0, v0, u1

def calibration(u0, v0, u1, u2, u4, w):
    f = np.sqrt(-(v0**2 + u0 * u1))
    phi = np.arctan(-v0 / f)
    theta = np.arctan(-u0 * np.cos(phi) / f)
    h = f * w * np.sin(phi) / (np.abs(u4 - u2) * np.cos(theta))

    return f, phi, theta, h


if __name__ == "__main__":
    # print(vanishing_points((786.819, 638.898), (594.339, 402.435), (683.12, 397.97), (923.054, 613.468)))
    # print(calibration(-620.5535005468671, -450.70218493772774, 5860.68444590289, -259.117, -125.941, 2.55))

    print(vanishing_points((947.638, 767.447), (709.228, 514.002), (806.093, 504.293), (1089.306, 736.378)))
    print(calibration(-616.4945560056715, -414.78434791685504, 3628.0815988223467, 728.476, 844.025, 2.55))
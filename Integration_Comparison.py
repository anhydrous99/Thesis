import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

yoshida_constants = {
    'c1': 0.6756035959798288170238,
    'c2': -0.1756035959798288170238,
    'c3': -0.1756035959798288170238,
    'c4': 0.6756035959798288170238,
    'd1': 1.3512071919596576340476,
    'd2': -1.7024143839193152680953,
    'd3': 1.3512071919596576340476
}

newmark_constants = {
    'alpha': 0.5,
    'beta': 0.25
}


def acc(r_input, GM=3.986004418E14, m=100):
    """
    Calculates the acceleration of the satellite w/ Drag

    :param r_input: Initial location
    :param GM: Standard Gravitational Parameter of the planet
    :param m: Mass of the satellite
    :return: The acceleration
    """
    r2 = np.dot(r_input, r_input)
    r = np.sqrt(r2)
    r_unit = r_input / r
    force = - (GM * m / r2) * r_unit
    a = force / m
    return a


def explicit_euler(r, v):
    """
    Calculates the next time step via explicit euler method

    :param r: Input position
    :param v: Input velocity
    :return: Resulting position and velocity
    """
    a = acc(r)
    r += v
    v += a
    return r, v


def semi_implicit_euler(r, v):
    """
    Calculates the next time step via the semi implicit euler method

    :param r: Input position
    :param v: Input velocity
    :return: Resulting position and velocity
    """
    a = acc(r)
    v += a
    r += v
    return r, v


def leapfrog2(r, v):
    """
    Calculates the next time step via the leap frog integration using the kick-drift-kick form

    :param r: Input position
    :param v: Input velocity
    :return: Resulting position and velocity
    """
    a = acc(r)
    v += 0.5 * a
    r += v
    v += 0.5 * acc(r)
    return r, v


def yoshida(r, v):
    """
    Calculates the next time step via the 4th order Yoshida integrator

    :param r: Input position
    :param v: Input velocity
    :return: Resulting position and velocity
    """
    r1 = r + yoshida_constants['c1'] * v
    v1 = v + yoshida_constants['d1'] * acc(r1)
    r2 = r1 + yoshida_constants['c2'] * v1
    v2 = v1 + yoshida_constants['d2'] * acc(r2)
    r3 = r2 + yoshida_constants['c3'] * v2
    v3 = v2 + yoshida_constants['d3'] * acc(r3)
    r = r3 + yoshida_constants['c4'] * v3
    return r, v3


def velocity_verlet(r, v):
    """
    Calculates the next time step via the velocity verlet integrator

    :param r: Input position
    :param v: Input velocity
    :return: Resulting position and velocity
    """
    a = acc(r)
    r += v + 0.5 * a
    v += 0.5 * (a + acc(r))
    return r, v


def rk4_helper(state):
    return np.array([state[1], acc(state[0])])


def rk4(r, v):
    """
    Calculates the next time step via the Classic 4th order Runge-Kutta method

    :param r: Input position
    :param v: Input velocity
    :return: Resulting position and velocity
    """
    state = np.array([r, v])
    k1 = rk4_helper(state)
    k2 = rk4_helper(state + k1 / 2)
    k3 = rk4_helper(state + k2 / 2)
    k4 = rk4_helper(state + k3)
    next_state = state + (1 / 6) * (k1 + 2 * (k2 + k3) + k4)
    return next_state[0], next_state[1]


def main():
    to_test = {
        'E Euler': explicit_euler,
        'SI Euler': semi_implicit_euler,
        'Leapfrog': leapfrog2,
        'Yoshida': yoshida,
        'Verlet': velocity_verlet,
        'rk4': rk4
    }
    data = {}
    for test in to_test:
        data[test] = []
        r = np.array([
            6921000,
            0
        ], dtype=np.float64)
        v = np.array([
            0,
            np.sqrt(3.986004418E14 / 6921000)
        ], dtype=np.float64)
        for _ in range(1000):
            data[test].append(np.abs(np.linalg.norm(r) - 6921000))
            r, v = to_test[test](r, v)
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 8), dpi=80)
    df.plot()
    plt.yscale('log')
    df.to_csv('data/Integration_comparison.csv')
    plt.savefig('plots/Integration_comparison.png', bbox_inches='tight')


if __name__ == '__main__':
    main()

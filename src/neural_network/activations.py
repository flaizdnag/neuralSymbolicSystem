import math


def idem(x: float) -> float:
    """
    :param x: any number (Float)
    :return: exactly the same number (Float)
    """
    return x


def idem_d(x: float) -> 1:
    return 1


setattr(idem, 'd', idem_d)


def const(x: float) -> float:
    """
    :param x: any number (Float)
    :return: 1 (Float)
    """
    return 1.0


def const_d(x: float) -> 0:
    return 0


setattr(const, 'd', const_d)


def sigm(x: float) -> float:
    """
    :param x: any number (Float)
    :return: Bipolar sigmoid function of x (Float)

    Note: this is not a hyperbolic tangent function. Its similar but slightly different
    """
    return (2 / (1 + math.exp(-x))) - 1


def sigm_d(x: float) -> float:
    """
    :param x: any number (Float)
    :return: Derivative of bipolar sigmoid function of x (Float)

    Note: this is not a derivative of hyperbolic tangent function. Its similar but slightly different.
    """
    return (2 * math.exp(-x)) / ((math.exp(-x) + 1) ** 2)


setattr(sigm, 'd', sigm_d)


def k(x, amin):
    return x

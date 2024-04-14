import numpy as np
import itertools

from plotly import graph_objects as go
from scipy.optimize import newton





def prng(s, a, m, it):
    x = s
    random_numbers = []
    for _ in range(it):
        x = (a * x) % m
        random_numbers.append(x / m)  # Normalize to [0, 1]
    return random_numbers


# Q1 - MC Integration

seed = 65539
def mlc_generator(modulus, a):
    """
    Multiplicative linear congruential generator.

    Parameters:
        modulus (int): The modulus value (m).
        a (int): The multiplier value (a).

    Returns:
        A pseudo-random number generator function.
    """
    def _generator():
        nonlocal a
        while True:
            a = (a * seed) % modulus
            yield a / modulus
    return _generator()

def mlcg(seed, a=1664525, m=2**32):
    """
    Multiplicative / Lehmer Linear Congruential Generator
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    seed: float
        Seed
    a: float
        Multiplier
        Defaults to ``1664525``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    float:
        Sequence of Pseudo-Random Numbers, having a period, that is sensitive to the choice of a and m.

    """
    while True:
        seed = (a * seed) % m
        yield seed

def mlcgList(N:int, range:tuple, seed:float=42, a:float=1664525, m:float=2**32):
    """
    Continuous
    Returns normalized list of MLCG-generated random values
    Uses values suggested by Numerical Recipes as default

    Parameters
    ----------
    N: int
        Number of random numbers to be generated
    range: tuple
        Range from where the numbers will be sampled
    seed: float
        Seed
        Defaults to ``42``
    a: float
        Multiplier
        Defaults to ``1664525``
    m: float
        Modulus
        Defaults to ``2**32``

    Returns
    -------
    numpy.ndarray
        Normalized list of MLCG-generated random values

    """    
    start, end = range

    rnList = np.array(list(itertools.islice(mlcg(seed, a, m), 0, N))) / m
    return end * rnList + start





import numpy as np
from scipy.stats import poisson


# TODO (Charles): Movielens specific.
def poisson_noise(num, noise_level):
    if np.random.randint(100) <= noise_level:
        return np.clip(poisson.rvs(num, size=1)[0], 1, 5)
    return num


# TODO (Charles): Movielens specific.
def gaussian_noise(num, noise_level):
    if np.random.randint(100) <= noise_level:
        mu = 0
        sigma = 1
        return np.clip(num + np.random.normal(mu, sigma, 1)[0], 1, 5)
    return num


# TODO (Charles): Movielens specific.
def gender_flipping(sex, noise_level):
    if np.random.randint(100) <= noise_level:
        if sex == 'M':
            return 'F'
        return 'M'
    return sex

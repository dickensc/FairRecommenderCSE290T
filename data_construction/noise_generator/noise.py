import numpy as np
import random
from scipy.stats import poisson

MALE_FAV_GENRE = ['action', 'adventure', 'thriller']
FEMALE_FAV_GENRE = ['romance', 'drama']


def label_poisson_noise(row, noise_level):
    if random.uniform(0, 1) <= noise_level:
        row['Rating'] = np.clip(poisson.rvs(row['Rating'], size=1)[0], 1, 5)
    return row


def label_gaussian_noise(row, noise_level):
    if random.uniform(0, 1) <= noise_level:
        mu = 0
        sigma = 1
        row['Rating'] = np.clip(row['Rating'] + np.random.normal(mu, sigma, 1)[0], 1, 5)
    return row


def gender_flipping(row, noise_level):
    if random.uniform(0, 1) <= noise_level:
        if row['Gender'] == 'M':
            row['Gender'] = 'F'
        else:
            row['Gender'] = 'M'
    return row


def genre_gender_label_noise(row, noise_level):
    if random.uniform(0, 1) <= noise_level:
        if row['Gender'] == 'M' and any(genre in row['Genres'].lower() for genre in MALE_FAV_GENRE):
            row['Rating'] = np.clip(row['Rating'] * 1.1, 1, 5)
        if row['Gender'] == 'F' and any(genre in row['Genres'].lower() for genre in FEMALE_FAV_GENRE):
            row['Rating'] = np.clip(row['Rating'] * 1.1, 1, 5)
    return row

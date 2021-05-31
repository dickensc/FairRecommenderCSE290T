import pandas as pd
import numpy as np
import os

from helpers import standardize_ratings
from noise_generator.generate_noisy_data import generate_noisy_data

from predicate_constructors.base_model_predicates.ratings import ratings_predicate
from predicate_constructors.base_model_predicates.rated import rated_predicate
from predicate_constructors.base_model_predicates.item import item_predicate
from predicate_constructors.base_model_predicates.user import user_predicate
from predicate_constructors.base_model_predicates.avg_item_rating import average_item_rating_predicate
from predicate_constructors.base_model_predicates.avg_user_rating import average_user_rating_predicate
from predicate_constructors.base_model_predicates.sim_content import sim_content_predicate
from predicate_constructors.base_model_predicates.sim_demo_users import sim_demo_users_predicate
from predicate_constructors.base_model_predicates.sim_items import sim_items_predicate
from predicate_constructors.base_model_predicates.sim_users import sim_users_predicate
from predicate_constructors.base_model_predicates.target import target_predicate
from predicate_constructors.base_model_predicates.nmf_ratings import nmf_ratings_predicate
from predicate_constructors.base_model_predicates.svd_ratings import svd_ratings_predicate
from predicate_constructors.base_model_predicates.nb_ratings import nb_ratings_predicate

from predicate_constructors.fairness_predicates.group import group_predicate
from predicate_constructors.fairness_predicates.group_1_avg_rating import group1_avg_rating_predicate
from predicate_constructors.fairness_predicates.group_2_avg_rating import group2_avg_rating_predicate
from predicate_constructors.fairness_predicates.group_1_item_rating import group1_item_rating_predicate
from predicate_constructors.fairness_predicates.group_2_item_rating import group2_item_rating_predicate
from predicate_constructors.fairness_predicates.group_1_genre_rating import group1_genre_rating_predicate
from predicate_constructors.fairness_predicates.group_2_genre_rating import group2_genre_rating_predicate
from predicate_constructors.fairness_predicates.group_1 import group_1
from predicate_constructors.fairness_predicates.group_2 import group_2
from predicate_constructors.fairness_predicates.is_genre import is_genre_predicate
from predicate_constructors.fairness_predicates.negative_prior import negative_prior
from predicate_constructors.fairness_predicates.positive_prior import positive_prior
from predicate_constructors.fairness_predicates.group_member import group_member_predicate
from predicate_constructors.fairness_predicates.group_avg_item_rating import group_average_item_rating_predicate
from predicate_constructors.fairness_predicates.group_avg_rating import group_average_rating_predicate
from predicate_constructors.fairness_predicates.group_item_block import group_item_block_predicate
from predicate_constructors.fairness_predicates.group_denominator import group_denominators
from predicate_constructors.fairness_predicates.constant import constant_predicate

DATA_PATH = "../psl-datasets/movielens/data"
N_FOLDS = 5

NOISE_LEVELS = {
    "poisson_noise": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "gaussian_noise": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "gender_flipping": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
}


def construct_movielens_predicates():
    """
    """

    """
    Create data directory to write output to
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    """
    Assuming that the raw data already exists in the data directory
    """
    (movies_df, ratings_df, user_df, gaussian_corrupted_ratings_dfs,
     poisson_corrupted_ratings_dfs, gender_flipping_user_dfs) = load_dataframes()
    (movies_df, ratings_df, user_df, gaussian_corrupted_ratings_dfs,
     poisson_corrupted_ratings_dfs, gender_flipping_user_dfs) = filter_dataframes(movies_df, ratings_df, user_df,
                                                                                  gaussian_corrupted_ratings_dfs,
                                                                                  poisson_corrupted_ratings_dfs,
                                                                                  gender_flipping_user_dfs)
    # note that truth and target will have the same atoms
    observed_ratings_df_list, train_ratings_df_list, test_ratings_df_list = partition_by_timestamp(ratings_df, N_FOLDS)
    # observed_ratings_df_list, train_ratings_df_list, test_ratings_df_list = partition_randomly(ratings_df, N_FOLDS)

    for fold, (observed_ratings_df, train_ratings_df, test_ratings_df) in \
            enumerate(zip(observed_ratings_df_list, train_ratings_df_list, test_ratings_df_list)):

        # # Standardized
        # # Learn
        # standardized_observed_ratings_df, standardized_truth_ratings_df = standardize_ratings(observed_ratings_df,
        #                                                                                       train_ratings_df)
        # write_predicates(standardized_observed_ratings_df, standardized_truth_ratings_df,
        #                  user_df, movies_df, 'learn', fold)
        #
        # # Eval
        # standardized_observed_ratings_df, standardized_truth_ratings_df = standardize_ratings(
        #     observed_ratings_df.append(train_ratings_df, verify_integrity=True), test_ratings_df)
        # write_predicates(standardized_observed_ratings_df, standardized_truth_ratings_df,
        #                  user_df, movies_df, 'eval', fold)

        # Un-standardized
        print("Fold: {} train predicates".format(fold))
        # Learn
        write_predicates(observed_ratings_df, train_ratings_df, gaussian_corrupted_ratings_dfs,
                         poisson_corrupted_ratings_dfs, user_df, gender_flipping_user_dfs,
                         movies_df, 'learn', fold)

        print("Fold: {} eval predicates".format(fold))
        print("Test Size: {}".format(test_ratings_df.shape[0]))
        # Eval
        write_predicates(observed_ratings_df.append(train_ratings_df, verify_integrity=True), test_ratings_df,
                         gaussian_corrupted_ratings_dfs, poisson_corrupted_ratings_dfs, user_df,
                         gender_flipping_user_dfs, movies_df, 'eval', fold)


def write_predicates(observed_ratings_df, truth_ratings_df, gaussian_corrupted_ratings_dfs, poisson_corrupted_ratings_dfs,
                     user_df, gender_flipping_user_dfs, movies_df, phase, fold):
    users = np.union1d(observed_ratings_df.index.get_level_values('userId').unique(),
                       truth_ratings_df.index.get_level_values('userId').unique())
    movies = np.union1d(observed_ratings_df.index.get_level_values('movieId').unique(),
                        truth_ratings_df.index.get_level_values('movieId').unique())
    movies_df = movies_df.loc[movies]
    user_df = user_df.loc[users]

    # Base model predicates

    # Ratings
    ratings_predicate(observed_ratings_df, partition='obs', fold=str(fold), phase=phase)
    ratings_predicate(truth_ratings_df, partition='targets', fold=str(fold), phase=phase, write_value=False)
    ratings_predicate(truth_ratings_df, partition='truth', fold=str(fold), phase=phase, write_value=True)

    for noise_level in NOISE_LEVELS["gaussian_noise"]:
        ratings_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                          partition='obs', fold=str(fold), phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))
        ratings_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[truth_ratings_df.index],
                          partition='targets', fold=str(fold), phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level),
                          write_value=False)
        ratings_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[truth_ratings_df.index],
                          partition='truth', fold=str(fold), phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level),
                          write_value=True)

    for noise_level in NOISE_LEVELS["poisson_noise"]:
        ratings_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                          partition='obs', fold=str(fold), phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))
        ratings_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[truth_ratings_df.index],
                          partition='targets', fold=str(fold), phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level),
                          write_value=False)
        ratings_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[truth_ratings_df.index],
                          partition='truth', fold=str(fold), phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level),
                          write_value=True)

    # Local Predictors
    nmf_ratings_predicate(observed_ratings_df, truth_ratings_df, fold=str(fold), phase=phase)
    svd_ratings_predicate(observed_ratings_df, truth_ratings_df, fold=str(fold), phase=phase)
    nb_ratings_predicate(observed_ratings_df, truth_ratings_df, user_df, movies_df, fold=str(fold), phase=phase)

    for noise_level in NOISE_LEVELS["gaussian_noise"]:
        nmf_ratings_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                              truth_ratings_df, fold=str(fold),
                              phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))
        svd_ratings_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                              truth_ratings_df, fold=str(fold),
                              phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))
        nb_ratings_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                             truth_ratings_df, user_df, movies_df, fold=str(fold),
                             phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))

    for noise_level in NOISE_LEVELS["poisson_noise"]:
        nmf_ratings_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                              truth_ratings_df, fold=str(fold),
                              phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))
        svd_ratings_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                              truth_ratings_df, fold=str(fold),
                              phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))
        nb_ratings_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                             truth_ratings_df, user_df, movies_df, fold=str(fold),
                             phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))

    for noise_level in NOISE_LEVELS["gender_flipping"]:
        nb_ratings_predicate(observed_ratings_df, truth_ratings_df,
                             gender_flipping_user_dfs[noise_level].loc[user_df.index],
                             movies_df, fold=str(fold),
                             phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    # Average
    average_item_rating_predicate(observed_ratings_df, fold=str(fold), phase=phase)
    average_user_rating_predicate(observed_ratings_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gaussian_noise"]:
        average_item_rating_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                                      fold=str(fold), phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))
        average_user_rating_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                                      fold=str(fold), phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))

    for noise_level in NOISE_LEVELS["poisson_noise"]:
        average_item_rating_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                                      fold=str(fold), phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))
        average_user_rating_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                                      fold=str(fold), phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))

    # Similarity
    sim_content_predicate(movies_df, fold=str(fold), phase=phase)

    sim_demo_users_predicate(user_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        sim_demo_users_predicate(gender_flipping_user_dfs[noise_level].loc[user_df.index], fold=str(fold),
                                 phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    sim_items_predicate(observed_ratings_df, movies, fold=str(fold), phase=phase)
    sim_users_predicate(observed_ratings_df, users, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gaussian_noise"]:
        sim_items_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index], movies,
                            fold=str(fold), phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))
        sim_users_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index], users,
                            fold=str(fold), phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))

    for noise_level in NOISE_LEVELS["poisson_noise"]:
        sim_items_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index], movies,
                            fold=str(fold), phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))
        sim_users_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index], users,
                            fold=str(fold), phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))

    # Scoping and Coarse Blocking
    item_predicate(observed_ratings_df, truth_ratings_df, fold=str(fold), phase=phase)
    user_predicate(observed_ratings_df, truth_ratings_df, fold=str(fold), phase=phase)
    rated_predicate(observed_ratings_df, truth_ratings_df, fold=str(fold), phase=phase)
    target_predicate(truth_ratings_df, fold=str(fold), phase=phase)

    # fairness intervention predicates
    group_predicate(user_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        group_predicate(gender_flipping_user_dfs[noise_level].loc[user_df.index], fold=str(fold),
                        phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    constant_predicate(fold=str(fold), phase=phase)
    negative_prior(fold=str(fold), phase=phase)
    positive_prior(fold=str(fold), phase=phase)
    group_member_predicate(user_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        group_member_predicate(gender_flipping_user_dfs[noise_level].loc[user_df.index], fold=str(fold),
                               phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    group_1(user_df, fold=str(fold), phase=phase)
    group_2(user_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        group_1(gender_flipping_user_dfs[noise_level].loc[user_df.index], fold=str(fold),
                phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))
        group_2(gender_flipping_user_dfs[noise_level].loc[user_df.index], fold=str(fold),
                phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    group1_avg_rating_predicate(fold=str(fold), phase=phase)
    group2_avg_rating_predicate(fold=str(fold), phase=phase)

    group_average_item_rating_predicate(observed_ratings_df, user_df, movies_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gaussian_noise"]:
        group_average_item_rating_predicate(gaussian_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                                            user_df, movies_df, fold=str(fold),
                                            phase="{}/{}/{}".format(phase, "label_gaussian_noise", noise_level))

    for noise_level in NOISE_LEVELS["poisson_noise"]:
        group_average_item_rating_predicate(poisson_corrupted_ratings_dfs[noise_level].loc[observed_ratings_df.index],
                                            user_df, movies_df, fold=str(fold),
                                            phase="{}/{}/{}".format(phase, "label_poisson_noise", noise_level))

    for noise_level in NOISE_LEVELS["gender_flipping"]:
        group_average_item_rating_predicate(observed_ratings_df, gender_flipping_user_dfs[noise_level].loc[user_df.index],
                                            movies_df, fold=str(fold),
                                            phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    group_average_rating_predicate(user_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        group_average_rating_predicate(gender_flipping_user_dfs[noise_level].loc[user_df.index], fold=str(fold),
                                       phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    group_item_block_predicate(user_df, truth_ratings_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        group_item_block_predicate(gender_flipping_user_dfs[noise_level].loc[user_df.index], truth_ratings_df,
                                   fold=str(fold), phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))

    group_denominators(user_df, truth_ratings_df, fold=str(fold), phase=phase)
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        group_denominators(gender_flipping_user_dfs[noise_level].loc[user_df.index], truth_ratings_df, fold=str(fold),
                           phase="{}/{}/{}".format(phase, "gender_flipping", noise_level))


def partition_randomly(ratings_df, n_folds, train_proportion=0.7):
    observed_ratings_df_list = []
    train_ratings_df_list = []
    test_ratings_df_list = []
    for fold_ratings_df in [ratings_df.sample(frac=1, random_state=i) for i in np.arange(n_folds)]:
        # train test split
        learn_split_ratings_df = fold_ratings_df.iloc[: int(fold_ratings_df.shape[0] * train_proportion), :]
        test_split_ratings_df = fold_ratings_df.iloc[int(fold_ratings_df.shape[0] * train_proportion):, :]

        # observed train split
        observed_split_ratings_df = learn_split_ratings_df.iloc[: int(learn_split_ratings_df.shape[0] * train_proportion), :]
        train_split_ratings_df = learn_split_ratings_df.iloc[int(learn_split_ratings_df.shape[0] * train_proportion):, :]

        observed_ratings_df_list.append(observed_split_ratings_df)
        train_ratings_df_list.append(train_split_ratings_df)
        test_ratings_df_list.append(test_split_ratings_df)

    return observed_ratings_df_list, train_ratings_df_list, test_ratings_df_list


def partition_by_timestamp(ratings_df, n_folds, train_proportion=0.7):
    sorted_frame = ratings_df.sort_values(by='timestamp')
    observed_ratings_df_list = []
    train_ratings_df_list = []
    test_ratings_df_list = []
    for fold_ratings_df in np.array_split(sorted_frame, n_folds, axis=0):
        # train test split
        learn_split_ratings_df = fold_ratings_df.iloc[: int(fold_ratings_df.shape[0] * train_proportion), :]
        test_split_ratings_df = fold_ratings_df.iloc[int(fold_ratings_df.shape[0] * train_proportion):, :]

        # observed train split
        observed_split_ratings_df = learn_split_ratings_df.iloc[: int(learn_split_ratings_df.shape[0] * train_proportion), :]
        train_split_ratings_df = learn_split_ratings_df.iloc[int(learn_split_ratings_df.shape[0] * train_proportion):, :]

        observed_ratings_df_list.append(observed_split_ratings_df)
        train_ratings_df_list.append(train_split_ratings_df)
        test_ratings_df_list.append(test_split_ratings_df)

    return observed_ratings_df_list, train_ratings_df_list, test_ratings_df_list


def filter_frame_by_group_rating(observed_ratings_df, truth_ratings_df, user_df):
    # filter movies not rated by both groups
    # TODO: Maybe we dont want to do this since this removes about 40% of movies and 7% of ratings.
    # However it circumvents the issue of calculating the average item rating for each group.
    truth_ratings_df_by_group = truth_ratings_df.groupby(lambda x: user_df.loc[truth_ratings_df.loc[x].userId].gender)
    movies_rated_by_group = truth_ratings_df_by_group['movieId'].unique()
    movies_rated_by_both_groups = set(movies_rated_by_group['F']).intersection(set(movies_rated_by_group['M']))
    filtered_truth_ratings_df = truth_ratings_df[truth_ratings_df.movieId.isin(movies_rated_by_both_groups)]
    return (observed_ratings_df[observed_ratings_df.movieId.isin(filtered_truth_ratings_df.movieId.unique())],
            filtered_truth_ratings_df)


def filter_dataframes(movies_df, ratings_df, user_df, gaussian_corrupted_ratings_dfs, poisson_corrupted_ratings_dfs,
                      gender_flipping_user_dfs, n=50, genres=None):
    """
    Preprocessing steps followed by Yao and Huang and Farnadi, Kouki, Thompson, Srinivasan, and  Getoor.
    Get rid of users who have not yet rated more than n movies.
    Remove movie that are not tagged with at least on of the genres: action romance crime musical and sci-fi.
    """
    if genres is None:
        genres = ['Action', 'Romance', 'Crime', 'Musical', 'Sci-Fi']

    # filter movies and ratings outside of the genres
    filtered_movies_df = movies_df[movies_df[genres].sum(axis=1) >= 1]
    filtered_ratings_df = ratings_df.reindex(filtered_movies_df.index, level='movieId').dropna(axis='index')

    # filter users that have less than n ratings
    filtered_ratings_df = filtered_ratings_df.groupby('userId').filter(lambda x: x.shape[0] > n)
    # filter ratings by users have dont have demographic information
    filtered_ratings_df = filtered_ratings_df.reindex(user_df.index, level='userId').dropna(axis='index')

    # filter users in user df that did not have n ratings
    filtered_user_df = user_df.loc[filtered_ratings_df.index.get_level_values('userId').unique()]
    # filter movies in movie df
    filtered_movies_df = filtered_movies_df.loc[filtered_ratings_df.index.get_level_values('movieId').unique()]

    # TODO: (Charles) Testing Purposes
    # filtered_ratings_df = filtered_ratings_df.sample(100)

    # Filer corresponding corrupted datasets.
    filtered_gaussian_corrupted_ratings_dfs = {}
    for noise_level in NOISE_LEVELS["gaussian_noise"]:
        filtered_gaussian_corrupted_ratings_dfs[noise_level] = gaussian_corrupted_ratings_dfs[noise_level].loc[filtered_ratings_df.index]

    filtered_poisson_corrupted_ratings_dfs = {}
    for noise_level in NOISE_LEVELS["poisson_noise"]:
        filtered_poisson_corrupted_ratings_dfs[noise_level] = poisson_corrupted_ratings_dfs[noise_level].loc[filtered_ratings_df.index]

    filtered_gender_flipping_user_dfs = {}
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        filtered_gender_flipping_user_dfs[noise_level] = gender_flipping_user_dfs[noise_level].loc[filtered_user_df.index]

    return (filtered_movies_df, filtered_ratings_df, filtered_user_df, filtered_gaussian_corrupted_ratings_dfs,
            filtered_poisson_corrupted_ratings_dfs, filtered_gender_flipping_user_dfs)


def load_dataframes():
    """
    Assuming that the raw data already exists in the data directory
    """
    movies_df = pd.read_csv(DATA_PATH + "/ml-1m/movies.dat", sep='::', header=None, encoding="ISO-8859-1",
                            engine='python')
    movies_df.columns = ["movieId", "movie title", "genres"]
    movies_df = movies_df.join(movies_df["genres"].str.get_dummies('|')).drop('genres', axis=1)
    movies_df = movies_df.astype({'movieId': int})
    movies_df = movies_df.set_index('movieId')

    ratings_df = pd.read_csv(DATA_PATH + '/ml-1m/ratings.dat', sep='::', header=None, engine='python')
    ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_df = ratings_df.astype({'userId': int, 'movieId': int})
    ratings_df.rating = ratings_df.rating / ratings_df.rating.max()
    ratings_df = ratings_df.set_index(['userId', 'movieId'])

    gaussian_corrupted_ratings_dfs = {}
    for noise_level in NOISE_LEVELS["gaussian_noise"]:
        gaussian_corrupted_ratings_dfs[noise_level] = pd.read_csv(
            DATA_PATH + '/ml-1m_label_gaussian_noise/{}/ratings.dat'.format(noise_level), sep='::', header=None, engine='python')
        gaussian_corrupted_ratings_dfs[noise_level].columns = ['userId', 'movieId', 'rating', 'timestamp']
        gaussian_corrupted_ratings_dfs[noise_level] = gaussian_corrupted_ratings_dfs[noise_level].astype({'userId': int, 'movieId': int})
        gaussian_corrupted_ratings_dfs[noise_level].rating = (
                gaussian_corrupted_ratings_dfs[noise_level].rating / gaussian_corrupted_ratings_dfs[noise_level].rating.max())
        gaussian_corrupted_ratings_dfs[noise_level] = gaussian_corrupted_ratings_dfs[noise_level].set_index(['userId', 'movieId'])

    poisson_corrupted_ratings_dfs = {}
    for noise_level in NOISE_LEVELS["poisson_noise"]:
        poisson_corrupted_ratings_dfs[noise_level] = pd.read_csv(
            DATA_PATH + '/ml-1m_label_poisson_noise/{}/ratings.dat'.format(noise_level), sep='::', header=None, engine='python')
        poisson_corrupted_ratings_dfs[noise_level].columns = ['userId', 'movieId', 'rating', 'timestamp']
        poisson_corrupted_ratings_dfs[noise_level] = poisson_corrupted_ratings_dfs[noise_level].astype({'userId': int, 'movieId': int})
        poisson_corrupted_ratings_dfs[noise_level].rating = (
                poisson_corrupted_ratings_dfs[noise_level].rating / poisson_corrupted_ratings_dfs[noise_level].rating.max())
        poisson_corrupted_ratings_dfs[noise_level] = poisson_corrupted_ratings_dfs[noise_level].set_index(['userId', 'movieId'])

    user_df = pd.read_csv(DATA_PATH + '/ml-1m/users.dat', sep='::', header=None,
                          encoding="ISO-8859-1", engine='python')
    user_df.columns = ['userId', 'gender', 'age', 'occupation', 'zip']
    user_df = user_df.astype({'userId': int})
    user_df = user_df.set_index('userId')

    gender_flipping_user_dfs = {}
    for noise_level in NOISE_LEVELS["gender_flipping"]:
        gender_flipping_user_dfs[noise_level] = pd.read_csv(
            DATA_PATH + '/ml-1m_gender_flipping/{}/users.dat'.format(noise_level), sep='::', header=None,
            encoding="ISO-8859-1", engine='python')
        gender_flipping_user_dfs[noise_level] .columns = ['userId', 'gender', 'age', 'occupation', 'zip']
        gender_flipping_user_dfs[noise_level] = gender_flipping_user_dfs[noise_level] .astype({'userId': int})
        gender_flipping_user_dfs[noise_level] = gender_flipping_user_dfs[noise_level] .set_index('userId')

    return movies_df, ratings_df, user_df, gaussian_corrupted_ratings_dfs, poisson_corrupted_ratings_dfs, gender_flipping_user_dfs


def main():
    generate_noisy_data()
    construct_movielens_predicates()


if __name__ == '__main__':
    main()

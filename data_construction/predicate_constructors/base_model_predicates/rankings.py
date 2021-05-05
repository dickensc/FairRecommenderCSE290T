import sys
sys.path.insert(0, '../..')
from helpers import write


def rankings_predicate(ratings_df, partition='obs', fold='0', phase='eval', write_value=True):
    """
    Ranking Predicates
    """
    rankings_series = ratings_df.loc[:, ['rating']]

    if write_value:
        write(rankings_series, 'ranking_' + partition, fold, phase)
    else:
        write(rankings_series.loc[:, []], 'ranking_' + partition, fold, phase)
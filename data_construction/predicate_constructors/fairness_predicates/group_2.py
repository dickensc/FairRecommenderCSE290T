import sys
sys.path.insert(0, '../..')

from helpers import write


def group_2(user_df, fold='0', phase='eval'):
    """
    group_2(U)
    """
    group_member_df = user_df.loc[:, ['gender']]
    group_member_df.loc[:, 'value'] = 0
    group_member_df[group_member_df.gender == 'M'].value = 1
    write(group_member_df.value, 'group_2_obs', fold, phase)
    write(group_member_df.loc[:, []], 'group_2_targets_targets', fold, phase)

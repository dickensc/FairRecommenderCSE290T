import sys
import os
import pandas as pd

# First read in GROUP_1 and GROUP_2 predictions.
# Round GROUP_1 and GROUP_2 predictions to 0, 1 values. Use maximum of the two for each user id.
# Write rounded predictions to same directory. Separate files for group_1 and group_2 in the same format as original file.
def main(out_directory):
    # Load attribute predictions.
    inferred_group_1_df = pd.read_csv(os.path.join(out_directory, "inferred-predicates", "GROUP_1_TARGETS.txt"), sep='\t', header=None)
    inferred_group_1_df.columns = ['user_id', 'value']
    inferred_group_1_df = inferred_group_1_df.set_index('user_id')

    inferred_group_2_df = pd.read_csv(os.path.join(out_directory, "inferred-predicates", "GROUP_2_TARGETS.txt"), sep='\t', header=None)
    inferred_group_2_df.columns = ['user_id', 'value']
    inferred_group_2_df = inferred_group_2_df.set_index('user_id')

    # Round attribute predictions
    rounded_group_1_df = inferred_group_1_df.reindex(inferred_group_2_df.index) > inferred_group_2_df
    rounded_group_1_df['value'] = rounded_group_1_df['value'].astype(int)
    rounded_group_2_df = inferred_group_2_df.reindex(inferred_group_1_df.index) > inferred_group_1_df
    rounded_group_2_df['value'] = rounded_group_2_df['value'].astype(int)

    # Write attribute predictions
    rounded_group_1_df.to_csv(os.path.join(out_directory, "inferred-predicates", "ROUNDED_GROUP_1_TARGETS.txt"), sep='\t', header=False)
    rounded_group_2_df.to_csv(os.path.join(out_directory, "inferred-predicates", "ROUNDED_GROUP_2_TARGETS.txt"), sep='\t', header=False)
    
    
def _load_args(args):
    if len(args) != 2 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 round_group_predictions <out_directory>", file=sys.stderr)
        sys.exit(1)
    else:
        return args[1]
    
    
if __name__ == '__main__':
    main(_load_args(sys.argv))
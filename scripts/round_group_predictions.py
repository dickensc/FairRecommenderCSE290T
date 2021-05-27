# First read in GROUP_1 and GROUP_2 predictions.
# Round GROUP_1 and GROUP_2 predictions to 0, 1 values. Use maximum of the two for each user id.
# Write rounded predictions to same directory. Separate files for group_1 and group_2 in the same format as original file.
def main():


def _load_args(args):
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
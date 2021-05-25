def main():


def _load_args(args):
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
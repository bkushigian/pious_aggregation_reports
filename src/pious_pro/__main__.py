from argparse import ArgumentParser, _SubParsersAction
from ._executables.version import VersionCliSubcommand
from ._executables.conf import ConfCliSubcommand
from ._executables.aggregate import AggregateCliSubcommand


def main():
    parser = ArgumentParser(
        prog="pious_pro",
        description="Advanced Utilities for PioSOLVER",
        allow_abbrev=True,
    )
    sp = parser.add_subparsers(title="commands")
    VersionCliSubcommand(sp)
    ConfCliSubcommand(sp)
    AggregateCliSubcommand(sp)

    args = parser.parse_args()
    if "function" in args:
        args.function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

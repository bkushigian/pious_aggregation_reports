from ..cli import CliSubcommand
from argparse import _SubParsersAction


class VersionCliSubcommand(CliSubcommand):
    def __init__(self, sub_parsers: _SubParsersAction):
        super().__init__(
            sub_parsers,
            "version",
            "Print Pious Pro version and exit",
        )

    def run(self, args) -> int:
        import importlib.metadata

        print(importlib.metadata.version("pious_pro"))
        return 0

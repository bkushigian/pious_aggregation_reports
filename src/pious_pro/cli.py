from argparse import Namespace, _SubParsersAction
from abc import ABC, abstractmethod


class CliSubcommand(ABC):
    """
    A generic class that should be extended by actual CLI Subcommands
    """

    def __init__(self, sub_parsers: _SubParsersAction, name: str, description: str):
        self.name = name
        self.description = description
        self.parser = sub_parsers.add_parser(
            name=self.name, description=self.description
        )
        self.parser.set_defaults(function=self)

    @abstractmethod
    def run(self, args: Namespace) -> int:
        raise NotImplementedError()

    def __call__(self, args: Namespace):
        exit(self.run(args))

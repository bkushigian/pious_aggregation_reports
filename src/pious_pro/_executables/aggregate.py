from ..cli import CliSubcommand
from pious.pio.aggregate import LinesToAggregate
from argparse import _SubParsersAction
from ..aggregate import run


class AggregateCliSubcommand(CliSubcommand):
    def __init__(self, sub_parsers: _SubParsersAction):
        super().__init__(
            sub_parsers,
            "aggregate",
            "Custom aggregation",
        )
        self.parser.add_argument(
            "cfr_file_or_sim_dir",
            default=".",
            help="Either a .cfr File or a folder containing .cfr files to aggregate",
        )
        self.parser.add_argument(
            "lines",
            nargs="*",
            help='Explicit nodes to add. These can be copied from PioViewer by right-clicking the board and selecting "Copy node id"',
        )
        self.parser.add_argument(
            "--flop", action="store_true", help="Add all flop nodes"
        )
        self.parser.add_argument(
            "--turn", action="store_true", help="Add all turn nodes"
        )
        self.parser.add_argument("--out", type=str, help="Directory to write files")
        self.parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrites an output directory if it exists",
        )
        self.parser.add_argument(
            "--progress", action="store_true", help="Print progress bar"
        )
        self.parser.add_argument(
            "--n_cores", type=int, default=1, help="Number of cores to use"
        )

    def run(self, args) -> int:
        try:
            run(args)
        except RuntimeError as e:
            print(e)
            return 1
        return 0

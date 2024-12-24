from ..cli import CliSubcommand
from argparse import _SubParsersAction


class ConfCliSubcommand(CliSubcommand):
    def __init__(self, sub_parsers: _SubParsersAction):
        super().__init__(
            sub_parsers,
            "conf",
            "Print Pious configuration and exit",
        )

    def run(self, args) -> int:
        from pious.conf import pious_conf
        from os import path as osp
        import importlib.metadata

        install_dir = pious_conf.pio_install_directory

        print(f'Pious Pro Version: {importlib.metadata.version("pious_pro")}')
        print(f'Pious Version: {importlib.metadata.version("pious")}')
        print(f"PioSOLVER:")
        print(
            f"    Install Directory          : {install_dir:32} EXISTS? {osp.exists(install_dir)}"
        )
        print(f"    PioSOLVER Version          : {pious_conf.pio_version_no}")
        print(f"    PioSOLVER Version Type     : {pious_conf.pio_version_type}")
        print(f"    PioSOLVER Version Suffix   : {pious_conf.pio_version_suffix}")
        pio_exec = osp.join(install_dir, pious_conf.get_pio_solver_name()) + ".exe"
        print(
            f"    PioSOLVER Executable       : {pio_exec:32} EXISTS? {osp.exists(pio_exec)}"
        )
        pio_viewer = osp.join(install_dir, pious_conf.get_pio_viewer_name()) + ".exe"
        print(
            f"    PioVIEWER                  : {pio_viewer:32} EXISTS? {osp.exists(pio_viewer)}"
        )

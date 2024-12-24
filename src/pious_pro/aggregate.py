from collections import defaultdict
from os import path as osp
from typing import List, Optional, Callable, Dict
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
import numpy as np

from pious.pio import Line, Node, Solver, make_solver
from pious.pio.aggregate import (
    LinesToAggregate,
    AggregationConfig,
    CFRDatabase,
    SpotData,
    collect_lines_to_aggregate,
    get_action_evs,
    get_action_freqs,
    get_action_names,
    get_sorted_actions,
)
from argparse import ArgumentParser, Namespace
from os import path as osp
import os
import shutil
import sys
from typing import Callable, Dict, List, Optional, Tuple
from pious.pio import Line, Node, make_solver
from pious.conf import pious_conf
import textwrap
import numpy as np
import pandas as pd
import time
from ansi.color import fg
from .excel import make_workbook_from_dict
from .trial import *
from .version import version, pious_version
import datetime


class FileAggregationResult:
    def __init__(self, board, cfr_file, lines_to_df: Dict[Line, pd.DataFrame]):
        self.board = board
        self.cfr_file = cfr_file
        self.lines_to_df = lines_to_df


class ParallelFileAggregator:
    def __init__(
        self,
        lines: List[Line],
        conf: AggregationConfig = None,
        conf_callback=None,
        print_progress=False,
    ):
        self.lines = lines
        self.conf = conf or AggregationConfig
        self.conf_callback = conf_callback
        self.print_progress = print_progress

    def aggregate_single_file(self, board, cfr_file, freq) -> FileAggregationResult:
        try:
            lines_to_df = aggregate_single_file(
                cfr_file,
                self.lines,
                self.conf,
                self.conf_callback,
                freq,
                self.print_progress,
                n_threads=1,
            )
            return FileAggregationResult(board, cfr_file, lines_to_df)
        except RuntimeError as e:
            print("Encountered error during aggregation on board", board)
            raise e

    def __call__(self, board, cfr_file, freq):
        return self.aggregate_single_file(board, cfr_file, freq)


def aggregate_files_in_dir(
    dir: str,
    lines: List[Line] | str | LinesToAggregate,
    conf: AggregationConfig = None,
    conf_callback: Optional[
        Callable[[Node, List[str], AggregationConfig], AggregationConfig]
    ] = None,
    print_progress: bool = False,
    n_threads: int = 1,
):
    if n_threads < 1:
        print(f"\033[33mWarning: `n_cores` set to {n_threads} < 1: setting to 1\033[0m")
    if conf is None:
        conf = AggregationConfig()

    db = CFRDatabase(dir)
    if len(db.cfr_files) == 0:
        raise RuntimeError(f"No CFR files found in {dir}")

    # We want to collect lines. To do this we need a solver instance with a tree
    # loaded, so we will grab the first cfr file in the DB
    cfr0 = osp.join(db.dir, db.cfr_files[0])
    solver: Solver = make_solver()
    solver.load_tree(cfr0)

    xs = db

    aggregator = ParallelFileAggregator(lines, conf, conf_callback, print_progress)
    results = None
    with Pool(n_threads) as pool:
        if print_progress:
            results: List[FileAggregationResult] = tqdm(
                pool.starmap(aggregator, db),
                total=len(db),
                ncols=150,
                desc="Boards",
            )
        else:
            results: List[FileAggregationResult] = pool.starmap(aggregator, xs)
    if results is None:
        print("\033[31;1mError\033[0m: Could not aggregate files")

    # Now combine results. We will use pd.concat, which takes an iterable of
    # dataframes. So for each line,
    combined = defaultdict(list)
    for r in results:
        for k, v in r.lines_to_df.items():
            combined[k].append(v)

    reports = {k: pd.concat(v, ignore_index=True) for k, v in combined.items()}

    return reports
    # for board, cfr_file, freq in xs:
    #     try:
    #         new_reports = aggregate_single_file(
    #             cfr_file, lines, conf, conf_callback, freq, print_progress, n_threads
    #         )

    #         # One time update: This is necessary to perform sanity checking and
    #         # ensure that Each report has the same lines.
    #         if reports is None:
    #             reports = new_reports
    #             reports_lines = set(reports.keys())
    #             continue

    #         # Perform Sanity Check
    #         new_reports_lines = set(new_reports.keys())
    #         if reports_lines != new_reports_lines:
    #             sym_diff = reports_lines.symmetric_difference(new_reports_lines)
    #             raise RuntimeError(
    #                 f"The following lines were not found in both reports: {sym_diff}"
    #             )

    #         # We know we have the same keyset, so combine the reports
    #         for line in new_reports:
    #             df1 = reports[line]
    #             df2 = new_reports[line]
    #             reports[line] = pd.concat([df1, df2], ignore_index=True)
    #     except RuntimeError as e:
    #         print("Encountered error during aggregation on board", board)
    #         raise e


def aggregate_single_file(
    cfr_file: str,
    lines: List[Line] | str | LinesToAggregate,
    conf: AggregationConfig = None,
    conf_callback: Optional[
        Callable[[Node, List[str], AggregationConfig], AggregationConfig]
    ] = None,
    weight: float = 1.0,
    print_progress: bool = False,
    n_threads: int = 1,
) -> Dict[Line, pd.DataFrame]:
    """
    Compute an aggregation report for the sim in `cfr_file` for each line in
    `lines_to_aggregate`
    """
    file_name: str = cfr_file
    assert osp.isfile(file_name)
    if not file_name.endswith(".cfr"):
        print(f"{file_name} must be a .cfr file")
        exit(-1)
    solver: Solver = make_solver()
    solver.load_tree(file_name)
    ls = LinesToAggregate.create_from(lines)
    lines_to_aggregate = collect_lines_to_aggregate(solver, ls)

    return aggregate_lines_for_solver(
        solver,
        lines_to_aggregate,
        conf,
        conf_callback,
        weight,
        print_progress,
        n_threads,
    )


def aggregate_line_for_solver(
    board,
    solver: Solver,
    line: Line,
    conf: Optional[AggregationConfig] = None,
    conf_callback: Optional[
        Callable[[Node, List[str], AggregationConfig], AggregationConfig]
    ] = None,
    weight: float = 1.0,
):
    try:
        node_ids = line.get_node_ids(dead_cards=board)

        # Get the first node_id to compute some global stuff about the line
        node_id = node_ids[0]
        actions = solver.show_children_actions(node_id)
        node: Node = solver.show_node(node_id)
        if conf_callback is not None:
            this_node_conf = conf_callback(node, actions, conf)
        else:
            this_node_conf = conf

        action_names = get_action_names(line, actions)

        # Compute columns
        columns = ["Flop", "Turn", "River"][: len(node.board) - 2]
        if this_node_conf.global_freq:
            columns.append("Global Freq")

        if this_node_conf.evs:
            columns.append("OOP EV")
            columns.append("IP EV")

        if this_node_conf.equities:
            columns.append("OOP Equity")
            columns.append("IP Equity")

        if this_node_conf.extra_columns is not None:
            for name, _ in this_node_conf.extra_columns:
                columns.append(name)

        sorted_actions = get_sorted_actions(actions)

        if this_node_conf.action_freqs:
            for a in sorted_actions:
                columns.append(f"{action_names[a]} Freq")
        if this_node_conf.action_evs:
            for a in sorted_actions:
                columns.append(f"{action_names[a]} EV")

        df = pd.DataFrame(columns=columns)

        for node_id in node_ids:
            spot_data = SpotData(solver, node_id)
            node: Node = solver.show_node(node_id)
            df.loc[len(df)] = compute_row(
                this_node_conf,
                spot_data,
                weight,
                actions,
                sorted_actions,
            )
        return df

    except RuntimeError as e:
        print(f"Encountered error aggregating line {line} on board {board}")
        raise e


class PoolAggregationContext:
    def __init__(
        self,
        cfr_file_path,
        board,
        conf: Optional[AggregationConfig],
        conf_callback,
        weight,
    ):
        self.cfr_file_path = cfr_file_path
        self.board = board
        self.conf = conf
        self.conf_callback = conf_callback
        self.weight = weight
        pass

    def __call__(self, line):
        s = make_solver()
        s.load_tree(self.cfr_file_path)
        df = aggregate_line_for_solver(
            self.board, s, line, self.conf, self.conf_callback, self.weight
        )
        return df


def aggregate_lines_for_solver(
    solver: Solver,
    lines_to_aggregate: List[Line],
    conf: Optional[AggregationConfig] = None,
    conf_callback: Optional[
        Callable[[Node, List[str], AggregationConfig], AggregationConfig]
    ] = None,
    weight: float = 1.0,
    print_progress: bool = False,
    n_threads: int = 1,
) -> Dict[Line, pd.DataFrame]:
    """
    Aggregate the lines for a `Solver` instance with a tree already loaded.

    :param solver: the `Solver` instance with a tree already loaded
    :param lines_to_aggregate: a `list` of `Line` instances that belong to the
    `Solver`'s loaded tree
    :param conf: an optional `AggregationConfig` to customize the aggregation
    :param weight: a weight to apply to the global frequency (e.g., if a flop is
    discounted for being less common)
    :param print_progress: print progress bar for aggregation
    """
    if conf is None:
        conf = AggregationConfig()
    board = solver.show_board().split()

    reports: Dict[Line, pd.DataFrame] = {}

    if n_threads <= 1:
        if print_progress:
            for line in tqdm(
                lines_to_aggregate,
                total=len(lines_to_aggregate),
                ncols=100,
                desc="Lines ",
            ):
                reports[line] = aggregate_line_for_solver(
                    board, solver, line, conf, conf_callback, weight
                )
        else:
            for line in lines_to_aggregate:
                reports[line] = aggregate_line_for_solver(
                    board, solver, line, conf, conf_callback, weight
                )
    else:

        ctx = PoolAggregationContext(
            solver.cfr_file_path, board, conf, conf_callback, weight
        )

        if print_progress:
            with Pool(processes=n_threads) as pool:
                results = tqdm(pool.map(ctx, lines_to_aggregate), ncols=100)
                return {line: df for (line, df) in zip(lines_to_aggregate, results)}
        else:
            with Pool(processes=n_threads) as pool:
                results = pool.map(ctx, lines_to_aggregate)
                return {line: df for (line, df) in zip(lines_to_aggregate, results)}

    return reports


def get_runout(solver: Solver, node_id: str) -> List[str]:
    node = solver.show_node(node_id)
    b = node.board
    flop = b[:3]
    row = ["".join(flop)]
    if len(b) > 3:
        row.append(b[3])
    if len(b) > 4:
        row.append(b[4])
    return row


def get_actions_to_strats(
    solver: Solver, node_id: str, actions: List[str]
) -> Dict[str, List[List[float]]]:
    strats_for_node = solver.show_strategy(node_id)
    actions_to_strats = {}
    for i, a in enumerate(actions):
        actions_to_strats[a] = np.array(strats_for_node[i])
    return actions_to_strats


def compute_row(
    conf: AggregationConfig,
    spot: SpotData,
    weight: float,
    actions: List[str],
    sorted_actions: List[str],
):
    node = spot.node
    node_id = node.node_id
    row = get_runout(spot.solver, node_id)

    if conf.global_freq:
        global_freq = spot.solver.calc_global_freq(node_id)
        row.append(global_freq * weight)

    action_to_strats = get_actions_to_strats(spot.solver, node_id, actions)

    if conf.evs:
        evs = [spot.ev(0), spot.ev(1)]
        row += evs

    if conf.equities:
        equities = [spot.eq(0), spot.eq(1)]
        row += equities

    if conf.extra_columns is not None:
        for _, fn in conf.extra_columns:
            r = fn(spot)
            row.append(r)

    # Compute Frequencies
    if conf.action_freqs:
        row += get_action_freqs(
            spot, node_id, spot.position, sorted_actions, action_to_strats
        )

    if conf.action_evs:
        row += get_action_evs(
            spot.solver,
            node_id,
            spot.position,
            sorted_actions,
            action_to_strats,
            spot._money_so_far[spot.node.get_position_idx()],
        )

    return row


banner = f"""
Create an aggregation report
"""


CACHING = True


def print_header():
    s = make_solver()
    print(f"PioSOLVER Install Location: {pious_conf.pio_install_directory}")
    print(f"PioSOLVER Version: {pious_conf.pio_version_no}")
    print(f"PioSOLVER Executable: {pious_conf.get_pio_solver_name()}")
    v = s._run("show_version")
    print(f"VERSION: {v}")


def main():
    global CACHING
    parser = ArgumentParser("aggregate")

    parser.add_argument(
        "cfr_file_or_sim_dir",
        default=".",
        help="Either a cfr file (for a single file aggregation report) or a directory containing cfr files",
    )
    parser.add_argument("lines", nargs="*", help="Explicit nodes to add")
    parser.add_argument("--flop", action="store_true", help="Add all flop nodes")
    parser.add_argument("--turn", action="store_true", help="Add all turn nodes")
    parser.add_argument("--out", type=str, help="Directory to write files")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite results of a computation"
    )
    parser.add_argument("--progress", action="store_true", help="Print progress bar")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores to use")
    args = parser.parse_args()

    run(args)


def run(args: Namespace):
    if not osp.exists(args.cfr_file_or_sim_dir):
        print(f"No such file or directory {args.cfr_file_or_sim_dir}")
        exit(-1)
    lines = LinesToAggregate(
        lines=args.lines,
        flop=args.flop,
        turn=args.turn,
        river=False,
    )

    reports = None
    # Begin by checking that we can write to output
    out_dir = None
    if args.out is not None:
        out_dir = osp.abspath(args.out)
        if osp.exists(out_dir) and not args.overwrite:
            print()
            print(f"\033[31mDestination exists!\033[0m")
            print()
            print(f"    \033[1m{out_dir}\033[0m")
            print()
            print(
                textwrap.fill(
                    f"Use \033[33m--overwrite\033[0m to overwrite existing directory, specify a new output directory with \033[33m--out NEW_DESTINATION\033[0m, or manually remove destination before rerunning.",
                    width=80,
                )
            )
            print()
            print("\033[1mExiting.\033[0m")
            print()
            sys.exit(1)

    conf = AggregationConfig()
    t0 = None
    t1 = None
    t0 = time.time()
    if osp.isdir(args.cfr_file_or_sim_dir):
        reports = aggregate_files_in_dir(
            args.cfr_file_or_sim_dir,
            lines,
            conf=conf,
            conf_callback=conf_callback,
            print_progress=args.progress,
            n_threads=args.n_cores,
        )
    elif osp.isfile(args.cfr_file_or_sim_dir):
        reports = aggregate_single_file(
            args.cfr_file_or_sim_dir,
            lines,
            conf=conf,
            conf_callback=conf_callback,
            print_progress=args.progress,
            n_threads=args.n_cores,
        )
    else:
        print(f"{args.cfr_file_or_sim_dir} is neither a .cfr file or a directory")
        exit(-1)

    t1 = time.time()
    # if args.print:
    #     for line in reports:
    #         print()
    #         print(f"----- {line} -----")
    #         df = reports[line]
    #         print(tabulate.tabulate(df, headers=df.keys()))
    #         print()
    print(f"Ran in {t1 - t0: 6.1f} seconds")
    out = args.out
    if out is None:
        out = get_out_dir()
    if out is not None and reports is not None:
        out_dir = osp.abspath(out)
        if osp.exists(out_dir):
            if args.overwrite:
                shutil.rmtree(out_dir)
            else:
                raise RuntimeError(f"Destination exists: {out_dir}")

        print("Creating dir", out_dir)
        os.makedirs(out_dir)
        workbooks = []
        write_metadata(out_dir, args, lines, t1 - t0)
        for line in reports:
            print(f"{fg.green(f"Making workbook for line {line}")}")
            df = reports[line]
            line_dir = make_line_directory(out_dir, line)
            sub_reports = get_sub_reports(df)
            for srn in sub_reports:
                sr = sub_reports[srn]
                csv_file_name = osp.join(line_dir, srn) + ".csv"
                sr.to_csv(csv_file_name, float_format="%.2f", index=False, na_rep="NaN")
            line_xlsx = osp.join(
                out_dir, line.line_str.replace("r:0", "r0").replace(":", "_") + ".xlsx"
            )
            wb = make_workbook_from_dict(sub_reports)
            workbooks.append((line_xlsx, wb))
        for line_xlsx, wb in workbooks:
            print(f"Saving workbook to {fg.blue(line_xlsx)}")
            wb.save(line_xlsx)


def write_metadata(out_dir: str, args: Namespace, lines, t: float):
    with open(osp.join(out_dir, "info.txt"), "w+") as f:
        f.write(f"file: {args.cfr_file_or_sim_dir}\n")
        f.write(f"n_cores: {args.n_cores}\n")
        f.write(f"pious_pro_version: {version()}\n")
        f.write(f"pious_version: {pious_version()}\n")
        f.write(f"lines: {lines.lines}\n")
        f.write(f"lines.flop: {lines.flop}\n")
        f.write(f"lines.turn: {lines.turn}\n")
        f.write(f"lines.river: {lines.river}\n")
        f.write(f"ran in: {t:5.1f} seconds")
        pass


def get_out_dir():
    current_datetime = datetime.datetime.now()
    directory_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return osp.join("pious_reports", directory_name)


def conf_callback(node: Node, actions: List[str], conf: AggregationConfig):

    new_conf = conf.copy()
    extra_columns = compute_extra_columns(node, actions)

    flat_categories = []
    for super_cat_name, sub_cats in extra_columns:
        for sub_cat_name, sub_cat in sub_cats:
            flat_categories.append((f"{super_cat_name}:{sub_cat_name}", sub_cat))
    new_conf.extra_columns = flat_categories
    return new_conf


def make_line_directory(out_dir, line: Line) -> str:
    """
    Create a new directory for the current
    """
    if isinstance(line, Line):
        line_str = line.line_str
    elif isinstance(line, str):
        line_str = line
    else:
        raise RuntimeError(f"Illegal line {line}: must be pious.pio.Line or str")
    line_str = line_str.replace("r:0", "r0").replace(":", "_")
    line_dir = osp.join(out_dir, line_str)
    os.makedirs(line_dir)
    return line_dir


def get_sub_reports(
    df: pd.DataFrame, add_right_columns=False
) -> Dict[str, pd.DataFrame]:
    """
    Collect sub reports by splitting column names on first ":"
    """
    unsplit_columns = []
    sub_reports: Dict[str, List[str]] = {}
    for column_name in df.columns:
        if ":" in column_name:
            key = column_name.split(":")[0]
            sub_reports.setdefault(key, [])
            sub_reports[key].append(column_name)
        else:
            unsplit_columns.append(column_name)

    left_columns = []
    right_columns = []

    def is_right_column(col):
        lower = col.lower()
        return "freq" in lower or "ev" in lower or "eq" in lower

    def is_left_column(col):
        return col.lower() in ("flop", "turn", "river")

    for col in unsplit_columns:
        if is_right_column(col):
            if add_right_columns:
                right_columns.append(col)
        elif is_left_column(col):
            left_columns.append(col)
        else:
            print("Skipping column from subreport:", col)

    # Now, map each sub_report to a dataframe
    result: Dict[str, pd.DataFrame] = {}
    result["Aggregation"] = df[unsplit_columns]
    for srn in sub_reports:
        column_names = sub_reports[srn]
        column_names = left_columns + column_names + right_columns

        df2 = df[column_names]
        df2.columns = df2.columns.str.removeprefix(srn + ":")
        df2 = df2.dropna(axis=1, how="all")
        result[srn] = df2
    return result


def action_freqs_closure(
    hand_query: Optional[str] = None,
    action_predicate: Callable = lambda a: a.startswith("b"),
    weight_type="MATCHUPS",
    cached_query=None,
):
    weight_type = weight_type.upper()

    if not CACHING:
        queries = []
        if hand_query is not None:
            queries.append(hand_query)
        if cached_query is not None:
            queries.append(cached_query)
        hand_query = " and ".join([f"({q})" for q in queries])
        cached_query = None

    def f(spot: SpotData):
        df = None  # Always initialized in then/else branches

        if cached_query is not None:
            df = spot._cache.get(cached_query, None)
            if df is None:
                # print(f"\033[31;1mCACHE_MISS\033[0m: {cached_query}")
                df = spot.hands_df().query(cached_query)
                spot._cache[cached_query] = df
            else:
                pass
                # print(f"\033[32;1mCACHE_HIT\033[0m: {cached_query}")
        else:
            df = spot.hands_df()
        if hand_query is not None:
            df = df.query(hand_query)
        actions = [a for a in spot.available_actions() if action_predicate(a)]
        action_freq_column_names = [a + "_freq" for a in actions]
        action_columns = df[action_freq_column_names].sum(axis=1)

        result = np.nan
        total_weights = 0.0
        if weight_type == "MATCHUPS":
            result = df["matchups"].dot(action_columns)
            total_weights = df["matchups"].sum()
        elif weight_type == "RANGE":
            result = df["range"].dot(action_columns)
            total_weights = df["range"].sum()
        else:
            print(f"Warning: unrecognized weight type {weight_type}, using 'MATCHUPS'")
            result = df["matchups"].dot(action_columns)
            total_weights = df["matchups"]

        if total_weights == 0.0:
            return np.nan
        return 100 * result / total_weights

    return f


def compute_extra_columns(node: Node, actions: List[str]):
    board = node.board
    can_fold = "f" in actions
    can_check_call = "c" in actions
    can_bet_raise = False
    for a in actions:
        if a.startswith("b"):
            can_bet_raise = True
            break

    measure_bdfd = False
    if len(board) == 3:
        measure_bdfd = True
    return get_hand_category_functions(
        board,
        actions=actions,
        measure_b=can_bet_raise,
        measure_c=can_check_call,
        measure_f=can_fold,
        measure_backdoors=measure_bdfd,
    )


def get_hand_category_functions(
    board,
    actions,
    measure_b=True,
    measure_c=False,
    measure_f=False,
    measure_backdoors=False,
) -> List[Tuple[str, List[Tuple[str, Callable]]]]:
    ranks = [c[0] for c in board]
    num_ranks = len(set(ranks))
    # Helper functions to determine what type of action we are looking at
    is_b_action = lambda a: a.startswith("b")
    is_c_action = lambda a: a == "c"
    is_f_action = lambda a: a == "f"

    # Give proper human-readable names to `b` and `c` actions
    b_str = "Bet"
    if "f" in actions:
        b_str = "Raise"

    c_str = "Check"
    if "f" in actions:
        c_str = "Call"

    wt = "MATCHUPS"

    # Define some helper functions to collect frequencies
    def bfreq(cached_query=None, query_str=None):
        return action_freqs_closure(query_str, is_b_action, wt, cached_query)

    def cfreq(cached_query=None, query_str=None):
        return action_freqs_closure(query_str, is_c_action, wt, cached_query)

    def ffreq(cached_query=None, query_str=None):
        return action_freqs_closure(query_str, is_f_action, wt, cached_query)

    hand_types = [
        "Nothing",
        "Pair",
        "TwoPair",
        "Trips",
        "Straight",
        "Flush",
        "FullHouse",
        "Quads",
        "StraightFlush",
    ]

    is_bdfd_str = "flush_type == '3_FLUSH'"
    is_bdsd_str = "straight_type == '3_STRAIGHT'"
    is_combo_str = "(flush_type == 'FLUSH_DRAW' and straight_type == 'STRAIGHT_DRAW')"
    is_fd_str = "flush_type == 'FLUSH_DRAW' and not straight_type == 'STRAIGHT_DRAW'"
    is_sd_str = "straight_type == 'STRAIGHT_DRAW' and not flush_type == 'FLUSH_DRAW'"

    def ht_freqs(ht_name, cached_query, query=None):
        results = []
        # bet/raise
        if measure_b:
            results.append((f"{ht_name}:{b_str}", bfreq(cached_query, query)))
        if measure_c:
            results.append((f"{ht_name}:{c_str}", cfreq(cached_query, query)))
        if measure_f:
            results.append((f"{ht_name}:Fold", ffreq(cached_query, query)))
        return results

    #############################
    # Compute Overview Category #
    #############################

    overview_cats = []

    for hand_type_idx, ht_name in enumerate(hand_types):
        ht_str = f"hand_type == {hand_type_idx}"
        overview_cats += ht_freqs(ht_name, ht_str)
    overview_cats += ht_freqs("Combo", is_combo_str)
    overview_cats += ht_freqs("FD", is_fd_str)
    overview_cats += ht_freqs("SD", is_sd_str)
    overview_cats += ht_freqs("BDFD", is_bdfd_str)

    ############################
    # Compute Nothing Category #
    ############################

    nothing_cats = []
    nothing_cats += ht_freqs("Nothing", "hand_type == 0")
    ht_str = "hand_type == 0"
    nothing_cats += ht_freqs("A-High:All", ht_str, "hr1 == 14")
    nothing_cats += ht_freqs("A-High:H", ht_str, "hr1 == 14 and hr2 >= 10")
    nothing_cats += ht_freqs("A-High:M", ht_str, "hr1 == 14 and hr2 < 10 and hr2 >= 6")
    nothing_cats += ht_freqs("A-High:L", ht_str, "hr1 == 14 and hr2 <= 5")

    nothing_cats += ht_freqs("K-High:All", ht_str, "hr1 == 13")
    nothing_cats += ht_freqs("K-High:H", ht_str, "hr1 == 13 and hr2 >= 10")
    nothing_cats += ht_freqs("K-High:M", ht_str, "hr1 == 13 and hr2 < 10 and hr2 >= 6")
    nothing_cats += ht_freqs("K-High:L", ht_str, "hr1 == 13 and hr2 <= 5")

    nothing_cats += ht_freqs("Q-High:All", ht_str, "hr1 == 12")
    nothing_cats += ht_freqs("Q-High:H", ht_str, "hr1 == 12 and hr2 >= 10")
    nothing_cats += ht_freqs("Q-High:M", ht_str, "hr1 == 12 and hr2 < 10 and hr2 >= 6")
    nothing_cats += ht_freqs("Q-High:L", ht_str, "hr1 == 12 and hr2 <= 5")

    nothing_cats += ht_freqs("J-High:All", ht_str, "hr1 == 11")
    nothing_cats += ht_freqs("T-High:All", ht_str, "hr1 == 10")
    for hand_type_idx in range(5):
        nothing_cats += ht_freqs(
            f"BoardGroup:[{hand_type_idx}]", f"high_card_1_type == {hand_type_idx}"
        )
    # Compute Draws
    nothing_cats += ht_freqs(f"Combo", ht_str, is_combo_str)
    nothing_cats += ht_freqs(f"FD", ht_str, is_fd_str)
    nothing_cats += ht_freqs(f"SD", ht_str, is_sd_str)
    if measure_backdoors:
        nothing_cats += ht_freqs(f"BDFD", ht_str, is_bdfd_str)

    # Overs and Unders
    over_1_ht = "high_card_1_type == 0"
    two_overs_str = "high_card_2_type == 0"
    under_2_ht = f"high_card_2_type >= {num_ranks}"

    # Two Over Cards
    nothing_cats += ht_freqs(f"TwoOvers:All", two_overs_str)
    if measure_backdoors:
        nothing_cats += ht_freqs(f"TwoOvers:BDFD", two_overs_str, is_bdfd_str)
        nothing_cats += ht_freqs(f"TwoOvers:BDSD", two_overs_str, is_bdsd_str)
    # Overs and Unders
    over_under = f"{over_1_ht} and {under_2_ht}"
    nothing_cats += ht_freqs(f"OverUnder:All", over_under)
    if measure_backdoors:
        nothing_cats += ht_freqs(f"OverUnder:BDFD", over_under, is_bdfd_str)
        nothing_cats += ht_freqs(f"OverUnder:BDSD", over_under, is_bdsd_str)

    #########################
    # Compute Pair Category #
    #########################

    pair_cats = []
    pair_names_and_queries = [
        ("OverPair", "pair_type == 0", "pair_cards_seen == 0"),
        ("TopPair", "pair_type == 2", "pair_cards_seen == 1"),
        ("UnderPair(1)", "pair_type == 0", "pair_cards_seen == 1"),
        ("2ndPair", "pair_type == 2", "pair_cards_seen == 2"),
        ("UnderPair(2)", "pair_type == 0", "pair_cards_seen == 2"),
        ("3rdPair", "pair_type == 2", "pair_cards_seen == 3"),
        ("UnderPair(3)", "pair_type == 0", "pair_cards_seen == 3"),
        ("4thPair", "pair_type == 2", "pair_cards_seen == 4"),
        ("UnderPair(4)", "pair_type == 0", "pair_cards_seen == 4"),
        ("5thPair", "pair_type == 2", "pair_cards_seen == 5"),
        ("UnderPair(5)", "pair_type == 0", "pair_cards_seen == 5"),
    ]
    for name, pair_type, pair_sub_query_str in pair_names_and_queries:
        pair_cats += ht_freqs(name, pair_type, pair_sub_query_str)

    # Now compute individual pair subsheets
    pair_subsheets = {}
    for name, pair_type, pair_sub_query_str in pair_names_and_queries:

        pair_subcats = []
        ht_str = f"{pair_type} and {pair_sub_query_str}"

        # Collect overall frequencies
        pair_subcats += ht_freqs(f"All", ht_str)

        # Collect per-kicker info
        if "underpair" in name.lower() or "overpair" in name.lower():
            pass
        else:
            for hand_type_idx in range(1, 12):
                kicker_str = f"pair_kicker == {hand_type_idx}"
                pair_subcats += ht_freqs(f"[{hand_type_idx}]", ht_str, kicker_str)

        pair_subcats += ht_freqs("FD", ht_str, is_fd_str)
        pair_subcats += ht_freqs("SD", ht_str, is_sd_str)
        if measure_backdoors:
            pair_subcats += ht_freqs("BDFD", ht_str, is_bdfd_str)
            pair_subcats += ht_freqs("BDSD", ht_str, is_bdsd_str)
        pair_subsheets[name] = pair_subcats

    categories = [
        ("Overview", overview_cats),
        ("Pairs", pair_cats),
    ]
    for pair_type, pair_subcats in pair_subsheets.items():
        categories.append((pair_type, pair_subcats))
    categories.append(("Nothing", nothing_cats))
    return categories


if __name__ == "__main__":
    main()

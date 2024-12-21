from collections import defaultdict
from os import path as osp
from typing import List, Optional, Callable, Dict
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
import numpy as np

from pious.pio import Line, Node, Solver, make_solver
from pious.progress_bar import progress_bar
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
            # print(board)
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
    #         # print(board)
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

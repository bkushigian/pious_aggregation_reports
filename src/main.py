from argparse import ArgumentParser
from os import path as osp
import os
import shutil
import sys
from typing import Callable, Dict, List, Optional, Tuple
import tabulate
from pious.pio import Line, Node
from pious.pio.aggregate import LinesToAggregate, AggregationConfig, SpotData
from aggregate import aggregate_files_in_dir, aggregate_single_file
from pious.hands import Hand
from pious.hand_categories import HandCategorizer
import textwrap
import numpy as np
import pandas as pd
import time

banner = f"""
Create an aggregation report
"""


CACHING = True


def main():
    global CACHING
    parser = ArgumentParser("aggregate")

    parser.add_argument(
        "cfr_file_or_sim_dir",
        help="Either a cfr file (for a single file aggregation report) or a directory containing cfr files",
    )
    parser.add_argument("lines", nargs="*", help="Explicit nodes to add")
    parser.add_argument("--flop", action="store_true", help="Add all flop nodes")
    parser.add_argument("--turn", action="store_true", help="Add all turn nodes")
    parser.add_argument("--river", action="store_true", help="Add all river nodes")
    parser.add_argument("--out", type=str, help="Directory to write files")
    parser.add_argument("--print", action="store_true", help="Print results to stdout")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite results of a computation"
    )
    parser.add_argument("--progress", action="store_true", help="Print progress bar")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores to use")
    parser.add_argument(
        "--time", action="store_true", help="Time the run of this program"
    )
    parser.add_argument(
        "--no_caching", action="store_true", help="Helper argument for testing"
    )
    args = parser.parse_args()
    if args.no_caching:
        CACHING = False

    if not osp.exists(args.cfr_file_or_sim_dir):
        print(f"No such file or directory {args.cfr_file_or_sim_dir}")
        exit(-1)

    lines = LinesToAggregate(
        lines=args.lines,
        flop=args.flop,
        turn=args.turn,
        river=args.river,
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
    if args.time:
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
        print(reports.keys())
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

    if args.time:
        t1 = time.time()
    if args.print:
        for line in reports:
            print()
            print(f"----- {line} -----")
            df = reports[line]
            print(tabulate.tabulate(df, headers=df.keys()))
            print()
    if args.time:
        print(f"Ran in {t1 - t0: 6.1f} seconds")
    if args.out is not None and reports is not None:
        out_dir = osp.abspath(args.out)
        if osp.exists(out_dir):
            if args.overwrite:
                shutil.rmtree(out_dir)
            else:
                raise RuntimeError(f"Destination exists: {out_dir}")

        print("Creating dir", out_dir)
        os.makedirs(out_dir)
        for line in reports:
            df = reports[line]
            line_dir = make_line_directory(out_dir, line)
            sub_reports = get_sub_reports(df)
            for srn in sub_reports:
                sr = sub_reports[srn]
                csv_file_name = osp.join(line_dir, srn) + ".csv"
                sr.to_csv(csv_file_name, float_format="%.2f", index=False, na_rep="NaN")


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
    print("outdir", out_dir)
    line_dir = osp.join(out_dir, line_str)
    print(f"Making line_dir: {line_dir}")
    os.makedirs(line_dir)
    return line_dir


def get_sub_reports(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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
    for col in unsplit_columns:
        if "freq" in col.lower():
            right_columns.append(col)
        elif "ev" in col.lower():
            right_columns.append(col)
        elif "eq" in col.lower():
            right_columns.append(col)
        elif col.lower() in ("flop", "turn", "river"):
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
    measure_bdfd = False
    if len(board) == 3:
        measure_bdfd = True
    return get_hand_category_functions(
        board,
        actions=actions,
        measure_c="f" in actions,
        measure_backdoors=measure_bdfd,
    )


def get_hand_category_functions(
    board,
    actions,
    measure_c=False,
    measure_backdoors=False,
) -> List[Tuple[str, List[Tuple[str, Callable]]]]:
    ranks = [c[0] for c in board]
    num_ranks = len(set(ranks))
    # Helper functions to determine what type of action we are looking at
    is_b_action = lambda a: a.startswith("b")
    is_c_action = lambda a: a == "c"

    # Give proper human-readable names to `b` and `c` actions
    b_str = "BetFreq"
    if "f" in actions:
        b_str = "RaiseFreq"

    c_str = "CheckFreq"
    if "f" in actions:
        c_str = "CallFreq"

    wt = "MATCHUPS"

    # Define some helper functions to collect frequencies
    def bfreq(cached_query=None, query_str=None):
        return action_freqs_closure(query_str, is_b_action, wt, cached_query)

    def cfreq(cached_query=None, query_str=None):
        return action_freqs_closure(query_str, is_c_action, wt, cached_query)

    hand_types = [
        "HighCard",
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
    is_fd_str = "flush_type == 'FLUSH_DRAW'"
    is_sd_str = "straight_type == 'STRAIGHT_DRAW'"

    #############################
    # Compute Overview Category #
    #############################

    overview_categories = []
    ovs = overview_categories

    # Compute Primary Hands
    for i, ht in enumerate(hand_types):
        ht_str = f"hand_type == {i}"
        ovs.append((f"{ht}:{b_str}", bfreq(ht_str)))
        if measure_c:
            ovs.append((f"{ht}:{c_str}", cfreq(ht_str)))
    # Compute Draws
    ovs.append((f"FlushDraw:{b_str}", bfreq(is_fd_str)))
    if measure_c:
        ovs.append((f"FlushDraw:{c_str}", cfreq(is_fd_str)))
    ovs.append((f"StraightDraw:{b_str}", bfreq(is_sd_str)))
    if measure_c:
        ovs.append((f"StraightDraw:{c_str}", cfreq(is_sd_str)))
    if measure_backdoors:
        ovs.append((f"BDFD:{b_str}", bfreq(is_bdfd_str)))
        if measure_c:
            ovs.append((f"BDFD:{c_str}", cfreq(is_bdfd_str)))

    ##############################
    # Compute High Card Category #
    ##############################

    high_card_subcategories = []
    hcs = high_card_subcategories
    for i in range(5):
        ht_str = f"high_card_1_type == {i}"
        hcs.append((f"[{i}]:{b_str}", bfreq(ht_str)))
        if measure_c:
            hcs.append((f"[{i}]:{c_str}", cfreq(ht_str)))
    # Compute Draws
    ht_str = "hand_type == 0"
    hcs.append((f"FlushDraw:{b_str}", bfreq(ht_str, is_fd_str)))
    if measure_c:
        hcs.append((f"FlushDraw:{c_str}", cfreq(ht_str, is_fd_str)))
    hcs.append((f"StraightDraw:{b_str}", bfreq(ht_str, is_sd_str)))
    if measure_c:
        hcs.append((f"StraightDraw:{c_str}", cfreq(ht_str, is_sd_str)))
    if measure_backdoors:
        # Back door flush draws
        hcs.append((f"BDFD:{b_str}", bfreq(ht_str, is_bdfd_str)))
        if measure_c:
            hcs.append((f"BDFD:{c_str}", cfreq(ht_str, is_bdfd_str)))
        # Back door straight draws
        hcs.append((f"BDSD:{b_str}", bfreq(ht_str, is_bdsd_str)))
        if measure_c:
            hcs.append((f"BDSD:{c_str}", cfreq(ht_str, is_bdsd_str)))
    # Overs and Unders
    over_1_ht = "high_card_1_type == 0"
    over_2_ht = "high_card_2_type == 0"
    under_2_ht = f"high_card_2_type >= {num_ranks}"
    hcs.append((f"TwoOvers:{b_str}", bfreq(over_2_ht)))
    if measure_c:
        hcs.append((f"TwoOvers:{c_str}", cfreq(over_2_ht)))
    if measure_backdoors:
        hcs.append((f"TwoOvers+BDFD:{b_str}", bfreq(over_2_ht, is_bdfd_str)))
        if measure_c:
            hcs.append((f"TwoOvers+BDFD:{c_str}", cfreq(over_2_ht, is_bdfd_str)))
        hcs.append((f"TwoOvers+BDSD:{b_str}", bfreq(over_2_ht, is_bdsd_str)))
        if measure_c:
            hcs.append((f"TwoOvers+BDSD:{c_str}", cfreq(over_2_ht, is_bdsd_str)))
    # Overs and Unders
    over_under = f"{over_1_ht} and {under_2_ht}"
    hcs.append((f"OverUnder:{b_str}", bfreq(over_under)))
    if measure_c:
        hcs.append((f"OverUnder:{c_str}", cfreq(over_under)))
    if measure_backdoors:
        hcs.append((f"OverUnder+BDFD:{b_str}", bfreq(over_under, is_bdfd_str)))
        if measure_c:
            hcs.append((f"TwoOvers+BDFD:{c_str}", cfreq(over_under, is_bdfd_str)))
        hcs.append((f"TwoOvers+BDSD:{b_str}", bfreq(over_under, is_bdsd_str)))
        if measure_c:
            hcs.append((f"TwoOvers+BDSD:{c_str}", cfreq(over_under, is_bdsd_str)))

    #########################
    # Compute Pair Category #
    #########################

    pair_subcategories = []
    names_and_queries = [
        ("OverPair", "pair_type == 0 and pair_cards_seen == 0"),
        ("TopPair", "pair_type == 2 and pair_cards_seen == 1"),
        ("UnderPair(1)", "pair_type == 0 and pair_cards_seen == 1"),
        ("2ndPair", "pair_type == 2 and pair_cards_seen == 2"),
        ("UnderPair(2)", "pair_type == 0 and pair_cards_seen == 2"),
        ("3rdPair", "pair_type == 2 and pair_cards_seen == 3"),
        ("UnderPair(3)", "pair_type == 0 and pair_cards_seen == 3"),
        ("4thPair", "pair_type == 2 and pair_cards_seen == 4"),
        ("UnderPair(4)", "pair_type == 0 and pair_cards_seen == 4"),
        ("5thPair", "pair_type == 2 and pair_cards_seen == 5"),
        ("UnderPair(5)", "pair_type == 0 and pair_cards_seen == 5"),
    ]
    for name, query_str in names_and_queries:
        pair_subcategories.append((f"{name}:{b_str}", bfreq(query_str)))
        if measure_c:
            pair_subcategories.append((f"{name}:{c_str}", cfreq(query_str)))

    #############################
    # Compute Top Pair Category #
    #############################

    top_pair_subcategories = []
    scs = top_pair_subcategories
    ht_str = "pair_type == 2 and pair_cards_seen == 1"
    for i in range(1, 12):
        kicker_str = f"pair_kicker == {i}"
        scs.append((f"[{i}]:{b_str}", bfreq(f"{ht_str} and {kicker_str}")))
        if measure_c:
            scs.append((f"[{i}]:{c_str}", cfreq(f"{ht_str} and {kicker_str}")))
    # Measure Top Pair Draws
    ## Flush Draws
    scs.append((f"FlushDraw:{b_str}", bfreq(f"{ht_str} and {is_fd_str}")))
    if measure_c:
        scs.append((f"FlushDraw:{c_str}", cfreq(f"{ht_str} and {is_fd_str}")))
    ## Straight Draws
    scs.append((f"StraightDraw:{b_str}", bfreq(f"{ht_str} and {is_sd_str}")))
    if measure_c:
        scs.append((f"StraightDraw:{c_str}", cfreq(f"{ht_str} and {is_sd_str}")))
    ## BDFDs
    if measure_backdoors:
        scs.append((f"BDFD:{b_str}", bfreq(f"{ht_str} and {is_bdfd_str}")))
        if measure_c:
            scs.append((f"BDFD:{c_str}", cfreq(f"{ht_str} and {is_bdfd_str}")))

    #############################
    # Compute 2nd Pair Category #
    #############################

    second_pair_subcategories = []
    scs = second_pair_subcategories
    ht_str = "pair_type == 2 and pair_cards_seen == 2"
    for i in range(1, 12):
        kicker_str = f"pair_kicker == {i}"
        scs.append((f"[{i}]:{b_str}", bfreq(f"{ht_str} and {kicker_str}")))
        if measure_c:
            scs.append((f"[{i}]:{c_str}", cfreq(f"{ht_str} and {kicker_str}")))
    # Measure Top Pair Draws
    ## Flush Draws
    scs.append((f"FlushDraw:{b_str}", bfreq(f"{ht_str} and {is_fd_str}")))
    if measure_c:
        scs.append((f"FlushDraw:{c_str}", cfreq(f"{ht_str} and {is_fd_str}")))
    ## Straight Draws
    scs.append((f"StraightDraw:{b_str}", bfreq(f"{ht_str} and {is_sd_str}")))
    if measure_c:
        scs.append((f"StraightDraw:{c_str}", cfreq(f"{ht_str} and {is_sd_str}")))
    ## BDFDs
    if measure_backdoors:
        scs.append((f"BDFD:{b_str}", bfreq(f"{ht_str} and {is_bdfd_str}")))
        if measure_c:
            scs.append((f"BDFD:{c_str}", cfreq(f"{ht_str} and {is_bdfd_str}")))

    #############################
    # Compute 3rd Pair Category #
    #############################

    third_pair_subcategories = []
    scs = third_pair_subcategories
    ht_str = "pair_type == 2 and pair_cards_seen == 3"
    for i in range(1, 12):
        kicker_str = f"pair_kicker == {i}"
        scs.append((f"[{i}]:{b_str}", bfreq(f"{ht_str} and {kicker_str}")))
        if measure_c:
            scs.append((f"[{i}]:{c_str}", cfreq(f"{ht_str} and {kicker_str}")))
    # Measure Top Pair Draws
    ## Flush Draws
    scs.append((f"FlushDraw:{b_str}", bfreq(f"{ht_str} and {is_fd_str}")))
    if measure_c:
        scs.append((f"FlushDraw:{c_str}", cfreq(f"{ht_str} and {is_fd_str}")))
    ## Straight Draws
    scs.append((f"StraightDraw:{b_str}", bfreq(f"{ht_str} and {is_sd_str}")))
    if measure_c:
        scs.append((f"StraightDraw:{c_str}", cfreq(f"{ht_str} and {is_sd_str}")))
    ## BDFDs
    if measure_backdoors:
        scs.append((f"BDFD:{b_str}", bfreq(f"{ht_str} and {is_bdfd_str}")))
        if measure_c:
            scs.append((f"BDFD:{c_str}", cfreq(f"{ht_str} and {is_bdfd_str}")))

    #############################
    # Compute 4rd Pair Category #
    #############################

    fourth_pair_subcategories = []
    scs = fourth_pair_subcategories
    ht_str = "pair_type == 2 and pair_cards_seen == 4"
    for i in range(1, 12):
        kicker_str = f"pair_kicker == {i}"
        scs.append((f"[{i}]:{b_str}", bfreq(f"{ht_str} and {kicker_str}")))
        if measure_c:
            scs.append((f"[{i}]:{c_str}", cfreq(f"{ht_str} and {kicker_str}")))
    # Measure Top Pair Draws
    ## Flush Draws
    scs.append((f"FlushDraw:{b_str}", bfreq(f"{ht_str} and {is_fd_str}")))
    if measure_c:
        scs.append((f"FlushDraw:{c_str}", cfreq(f"{ht_str} and {is_fd_str}")))
    ## Straight Draws
    scs.append((f"StraightDraw:{b_str}", bfreq(f"{ht_str} and {is_sd_str}")))
    if measure_c:
        scs.append((f"StraightDraw:{c_str}", cfreq(f"{ht_str} and {is_sd_str}")))
    ## BDFDs
    if measure_backdoors:
        scs.append((f"BDFD:{b_str}", bfreq(f"{ht_str} and {is_bdfd_str}")))
        if measure_c:
            scs.append((f"BDFD:{c_str}", cfreq(f"{ht_str} and {is_bdfd_str}")))

    return [
        ("Overview", overview_categories),
        ("Pairs", pair_subcategories),
        ("TopPair", top_pair_subcategories),
        ("2ndPair", second_pair_subcategories),
        ("3rdPair", third_pair_subcategories),
        ("4thPair", fourth_pair_subcategories),
        ("High", high_card_subcategories),
    ]


if __name__ == "__main__":
    main()

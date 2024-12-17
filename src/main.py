from argparse import Namespace, ArgumentParser
from os import path as osp
import os
import shutil
import sys
from typing import Callable, Dict, List, Tuple
import tabulate
from pious.pio import aggregate, Line, Node
from pious.hands import Hand
from pious.hand_categories import HandCategorizer
import textwrap
import numpy as np
import pandas as pd

banner = f"""
Create an aggregation report
"""


def main():
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
    args = parser.parse_args()

    if not osp.exists(args.cfr_file_or_sim_dir):
        print(f"No such file or directory {args.cfr_file_or_sim_dir}")
        exit(-1)

    lines = aggregate.LinesToAggregate(
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

    hand_categories = get_hand_category_functions()
    flat_categories = []
    for super_cat_name, sub_cats in hand_categories:
        for sub_cat_name, sub_cat in sub_cats:
            flat_categories.append((f"{super_cat_name}:{sub_cat_name}", sub_cat))

    conf = aggregate.AggregationConfig(extra_columns=flat_categories)
    if osp.isdir(args.cfr_file_or_sim_dir):
        reports = aggregate.aggregate_files_in_dir(
            args.cfr_file_or_sim_dir,
            lines,
            conf=conf,
            conf_callback=conf_callback,
            print_progress=args.progress,
        )
        print(reports.keys())
    elif osp.isfile(args.cfr_file_or_sim_dir):
        reports = aggregate.aggregate_single_file(
            args.cfr_file_or_sim_dir,
            lines,
            conf=conf,
            conf_callback=conf_callback,
            print_progress=args.progress,
        )
        pass
    else:
        print(f"{args.cfr_file_or_sim_dir} is neither a .cfr file or a directory")
        exit(-1)

    if args.print:
        for line in reports:
            print()
            print(f"----- {line} -----")
            df = reports[line]
            print(tabulate.tabulate(df, headers=df.keys()))
            print()
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


def conf_callback(node: Node, conf: aggregate.AggregationConfig):
    return conf


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


def create_compute_action_freqs_closure(
    hand_query: str,
    action_predicate: Callable = lambda a: a.startswith("b"),
    weight_type="MATCHUPS",
):
    weight_type = weight_type.upper()

    def f(spot: aggregate.SpotData):
        df = spot.hands_df().query(hand_query)
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


def is_overpair(h: Hand):
    if not h.is_pair():
        return False
    (pair_type, board_cards_seen, _) = HandCategorizer.get_pair_category(h)
    return pair_type == HandCategorizer.POCKET_PAIR and board_cards_seen == 0


def is_underpair(h: Hand, strength):
    if not h.is_pair():
        return False
    (pair_type, board_cards_seen, _) = HandCategorizer.get_pair_category(h)
    return pair_type == HandCategorizer.POCKET_PAIR and board_cards_seen == strength


def is_pair_with_qualities(h: Hand, pair_strength, kicker_value=None):
    if not h.is_pair():
        return False
    (pair_type, board_cards_seen, kicker) = HandCategorizer.get_pair_category(h)
    return (
        pair_type == HandCategorizer.REGULAR_PAIR
        and board_cards_seen == pair_strength
        and (kicker_value is None or kicker_value == kicker)
    )


def is_high_card_with_qualities(
    h: Hand, p_top_card, p_bottom_card=None, kicker_value=None
):
    if not h.is_high_card():
        return False
    try:
        (top, bottom) = HandCategorizer.get_high_card_category(h)
    except Exception as e:
        print(h)
        raise e
    return (p_top_card is None or p_top_card == top) and (
        p_bottom_card is None or p_bottom_card == bottom
    )


def get_hand_category_functions() -> List[Tuple[str, List[Tuple[str, Callable]]]]:
    def cc(predicate):
        return create_compute_action_freqs_closure(predicate, is_bet, wt)

    is_bet = lambda a: a.startswith("b")
    wt = "MATCHUPS"

    basic_categories = [
        ("HighCard", cc("hand_type == 0")),
        ("Pair", cc("hand_type == 1")),
        ("TwoPair", cc("hand_type == 2")),
        ("Trips", cc("hand_type == 3")),
        ("Straight", cc("hand_type == 4")),
        ("Flush", cc("hand_type == 5")),
        ("FullHouse", cc("hand_type == 6")),
        ("Quads", cc("hand_type == 7")),
        ("StraightFlush", cc("hand_type == 8")),
    ]

    pair_subcategories = [
        ("OverPair", cc("pair_type == 0 and pair_cards_seen == 0")),
        ("TopPair", cc("pair_type == 2 and pair_cards_seen == 1")),
        ("UnderPair(1)", cc("pair_type == 0 and pair_cards_seen == 1")),
        ("2ndPair", cc("pair_type == 2 and pair_cards_seen == 2")),
        ("UnderPair(2)", cc("pair_type == 0 and pair_cards_seen == 2")),
        ("3rdPair", cc("pair_type == 2 and pair_cards_seen == 3")),
        ("UnderPair(3)", cc("pair_type == 0 and pair_cards_seen == 3")),
        ("4thPair", cc("pair_type == 2 and pair_cards_seen == 4")),
        ("UnderPair(4)", cc("pair_type == 0 and pair_cards_seen == 4")),
        ("5thPair", cc("pair_type == 2 and pair_cards_seen == 5")),
        ("UnderPair(5)", cc("pair_type == 0 and pair_cards_seen == 5")),
    ]

    top_pair_subcategories = [
        ("[1]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 1")),
        ("[2]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 2")),
        ("[3]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 3")),
        ("[4]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 4")),
        ("[5]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 5")),
        ("[6]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 6")),
        ("[7]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 7")),
        ("[8]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 8")),
        ("[9]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 9")),
        ("[10]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 10")),
        ("[11]", cc("pair_type == 2 and pair_cards_seen == 1 and pair_kicker == 11")),
    ]

    second_pair_subcategories = [
        ("[1]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 1")),
        ("[2]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 2")),
        ("[3]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 3")),
        ("[4]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 4")),
        ("[5]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 5")),
        ("[6]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 6")),
        ("[7]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 7")),
        ("[8]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 8")),
        ("[9]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 9")),
        ("[10]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 10")),
        ("[11]", cc("pair_type == 2 and pair_cards_seen == 2 and pair_kicker == 11")),
    ]

    third_pair_subcategories = [
        ("[1]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 1")),
        ("[2]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 2")),
        ("[3]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 3")),
        ("[4]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 4")),
        ("[5]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 5")),
        ("[6]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 6")),
        ("[7]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 7")),
        ("[8]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 8")),
        ("[9]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 9")),
        ("[10]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 10")),
        ("[11]", cc("pair_type == 2 and pair_cards_seen == 3 and pair_kicker == 11")),
    ]

    fourth_pair_subcategories = [
        ("[1]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 1")),
        ("[2]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 2")),
        ("[3]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 3")),
        ("[4]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 4")),
        ("[5]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 5")),
        ("[6]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 6")),
        ("[7]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 7")),
        ("[8]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 8")),
        ("[9]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 9")),
        ("[10]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 10")),
        ("[11]", cc("pair_type == 2 and pair_cards_seen == 4 and pair_kicker == 11")),
    ]

    high_card_subcategories = [
        ("[0]", cc("high_card_1_type == 0")),
        ("[1]", cc("high_card_1_type == 1")),
        ("[2]", cc("high_card_1_type == 2")),
        ("[3]", cc("high_card_1_type == 3")),
        ("[4]", cc("high_card_1_type == 4")),
        ("FlushDraw", cc("hand_type == 0 and flush_type == 'FLUSH_DRAW'")),
    ]

    return [
        ("Overview", basic_categories),
        ("Pairs", pair_subcategories),
        ("TopPair", top_pair_subcategories),
        ("2ndPair", second_pair_subcategories),
        ("3rdPair", third_pair_subcategories),
        ("4thPair", fourth_pair_subcategories),
        ("High", high_card_subcategories),
    ]


if __name__ == "__main__":
    main()

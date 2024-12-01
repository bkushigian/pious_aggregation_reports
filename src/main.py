from argparse import Namespace, ArgumentParser
from os import path as osp
import os
import shutil
import sys
from typing import Callable, List, Tuple
import tabulate
from pious.pio import aggregate
import textwrap
import numpy as np

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
    conf = aggregate.AggregationConfig(extra_columns=hand_categories)
    if osp.isdir(args.cfr_file_or_sim_dir):
        reports = aggregate.aggregate_files_in_dir(
            args.cfr_file_or_sim_dir, lines, conf=conf
        )
        print(reports.keys())
    elif osp.isfile(args.cfr_file_or_sim_dir):
        reports = aggregate.aggregate_single_file(
            args.cfr_file_or_sim_dir, lines, conf=conf
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
            csv_file_name = (
                osp.join(out_dir, line.line_str.replace("r:0:", "").replace(":", "_"))
                + ".csv"
            )
            print(csv_file_name)
            df.to_csv(csv_file_name, float_format="%.2f", index=False, na_rep="NaN")


def create_compute_action_freqs_closure(
    action_predicate: Callable = lambda a: a.startswith("b"),
    hand_filter: Callable = lambda hd: hd.is_pair(),
    weight_type="MATCHUPS",
):
    weight_type = weight_type.upper()

    def f(spot: aggregate.SpotData):
        pos_idx = spot.node.get_position_idx()
        hds = spot.hand_details(pos_idx=spot.node.get_position_idx())
        strat = spot.strategy()
        actions = spot.available_actions()

        # Each entry is a betting action along with the frequencies of each hand
        # taking that action
        bet_freqs = [freqs for (a, freqs) in zip(actions, strat) if action_predicate(a)]
        total_weights = 0.0

        if weight_type == "MATCHUPS":
            weights = spot.matchups(pos_idx)
        elif weight_type == "RANGE":
            weights = spot.range(pos_idx).range_array
        else:
            print(f"Warning: unrecognized weight type {weight_type}, using 'MATCHUPS'")
            weights = spot.matchups(pos_idx)

        # Iterate over every hand and accumulate the total (unnormalized)
        # frequencies of taking that action (this will be normalized at return)
        filtered_action_freq = 0.0
        for hand_idx, hd in enumerate(hds):
            if hd is None or not hand_filter(hd):
                continue

            # Compute how much this hand contributes to overall frequency (mus)
            # and update the total number of matchups for all pairs
            hand_weight = weights[hand_idx]
            total_weights += hand_weight

            # Measure the frequency of the desired action set
            hand_bet_freq = sum([freqs[hand_idx] for freqs in bet_freqs])
            filtered_action_freq += hand_bet_freq * hand_weight

        if total_weights == 0.0:
            return np.nan
        return 100 * filtered_action_freq / total_weights

    return f


def get_hand_category_functions() -> List[Tuple[str, Callable]]:

    is_bet = lambda a: a.startswith("b")
    is_high_card = lambda h: h.is_high_card()
    is_pair = lambda h: h.is_pair()
    is_2pair = lambda h: h.is_two_pair()
    is_trips = lambda h: h.is_trips()
    is_straight = lambda h: h.is_straight()
    is_flush = lambda h: h.is_flush()
    is_full_house = lambda h: h.is_full_house()
    is_quads = lambda h: h.is_quads()
    is_straight_flush = lambda h: h.is_straight_flush()
    wt = "MATCHUPS"

    create_closure = create_compute_action_freqs_closure
    return [
        ("HighCard", create_closure(is_bet, is_high_card, wt)),
        ("Pair", create_closure(is_bet, is_pair, wt)),
        ("TwoPair", create_closure(is_bet, is_2pair, wt)),
        ("Trips", create_closure(is_bet, is_trips, wt)),
        ("Straight", create_closure(is_bet, is_straight, wt)),
        ("Flush", create_closure(is_bet, is_flush, wt)),
        ("FullHouse", create_closure(is_bet, is_full_house, wt)),
        ("Quads", create_closure(is_bet, is_quads, wt)),
        ("StraightFlush", create_closure(is_bet, is_straight_flush, wt)),
    ]


if __name__ == "__main__":
    main()

from edge_selection_tree import evaluate_edge_list, evaluate_edge_list_exact
from graphstructure import GraphStructure
import argparse
import numpy as np
import itertools
from tqdm import tqdm as tqdm
from pathlib import Path
import pandas as pd

from greedy_edge_choice import greedily_choose_edge_list_timelimit
from utils import add_uniform_probabilities, initialize_random_edge_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--p", type=float, default=0.01)
parser.add_argument("--num-vertices", type=int, default=75)
parser.add_argument("--chain-cap", type=int, default=4)
parser.add_argument("--cycle-cap", type=int, default=3)
parser.add_argument("--edge-budget", type=int, default=3)
parser.add_argument("--time-limit", type=int, default=1000)
parser.add_argument("--num-graphs", type=int, default=100)
parser.add_argument("--output-file", type=str, default="greedy_opt_output.txt")


if __name__ == "__main__":
    args = parser.parse_args()
    if Path(args.output_file).exists():
        print("warning, output file already exists")
    rs_greedy = np.random.RandomState(args.seed)
    edge_selection_rs = np.random.RandomState(args.seed)
    rg_seed = args.seed
    for i in range(args.num_graphs):
        randomgraph = GraphStructure.er_randomgraph(
            args.num_vertices,
            args.p,
            rg_seed,
            cycle_cap=args.cycle_cap,
            chain_cap=args.chain_cap,
        )
        rg_seed += 1
        initialize_random_edge_weights(randomgraph, rs_greedy)
        if len(randomgraph.matchable_edge_list) < args.edge_budget:
            print("skipping graph; matchable edges < edge budget")
            continue
        greedy_edges = greedily_choose_edge_list_timelimit(
            randomgraph,
            args.edge_budget,
            rs_greedy,
            args.time_limit,
            128,  # num_leaf_samples
            8,  # max_level_for_pruning
        )

        weight_list, prob_list, _ = evaluate_edge_list_exact(
            greedy_edges, randomgraph
        )
        if sum(prob_list) == 1.0:
            greedy_value = np.dot(weight_list, prob_list)
        else:
            greedy_value = np.mean(weight_list)

        best_subset = None
        best_value = 0.0
        for possible_subset in tqdm(
            list(
                itertools.combinations(
                    randomgraph.all_edge_list, args.edge_budget
                )
            )
        ):

            weight_list, prob_list, _ = evaluate_edge_list_exact(
                list(possible_subset), randomgraph
            )

            if sum(prob_list) == 1.0:
                # if probs sum to 1, this is exhaustive, exactly compute expectation
                subset_value = np.dot(weight_list, prob_list)
            else:
                # otherwise take sample mean
                subset_value = np.mean(weight_list)
            if subset_value > best_value:
                best_value = subset_value
                best_subset = list(possible_subset)


        greedy_choices = [str(e) for e in greedy_edges]
        best_choices = [str(e) for e in best_subset]
        print(set(greedy_edges) == set(best_subset))
        if greedy_value > best_value + 1e-6:
            print("greedy better???", i)
        ratio = float((greedy_value - best_value) / best_value)
        with open(args.output_file, "a") as f:
            out_str = (
                ",".join(greedy_choices)
                + ";"
                + ",".join(best_choices)
                + ";"
                + str(greedy_value)
                + ";"
                + str(best_value)
                + ";" + str(ratio)
            )
            print(out_str, file=f)

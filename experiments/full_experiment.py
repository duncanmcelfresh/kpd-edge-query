# function for running experiments with different robust kidney exchange formulations, in both the single-stage and
# multi-stage setting

import argparse
import logging
import os
import time
from collections import namedtuple

import numpy as np

from edge_selection_tree import (
    evaluate_edge_list,
    evaluate_accept_reject_outcome,
    evaluate_edge_list_with_data,
    create_edge_subset_search_tree,
    eval_sol_on_outcome,
    EdgeSubsetNode,
)
from graphstructure import GraphStructure
from greedy_edge_choice import (
    ignorance_is_almost_bliss,
    greedily_choose_edge_list_timelimit,
    create_greedy_policy,
)
from kidney_graph_io import (
    get_cmu_graphs,
    get_unos_graphs,
    get_single_cmu_graph,
    read_unos_graph_with_data,
)
from kidney_ip import solve_picef_model
from kpd_data import KPDData
from multistage_edge_selection import create_multistage_search_tree, create_mcts_policy
from utils import (
    generate_filepath,
    expected_matching_weight_noquery,
    get_matching_kpd_data,
    get_logger,
    LOG_FORMAT,
    add_uniform_probabilities,
    add_kpd_probabilities,
)

logger = get_logger()

DELIMITER = ";"
NULL_OUTPUT = None


def simulate_policy_multistage(policy_fn, graph, edge_rejections, num_stages, tree):
    """
    Takes a policy function of form policy_fn(graph, query_seq, outcome_seq) -> next_edge
    and evaluates it, given some presampled edge outcomes and a graph.
    """
    queried_edges = []
    edge_outcomes = []
    for stage in range(num_stages):
        next_edge = policy_fn(graph, queried_edges, edge_outcomes, tree)
        if next_edge is not None:
            queried_edges.append(next_edge)
            edge_outcomes.append(edge_rejections[next_edge.index])
        else:
            logger.info(
                f"multistage policy ran out of edges during stage {stage}. returning {len(queried_edges)} edges."
            )
            break

    return queried_edges


def sample_edge_rejections(graph, rs):
    rejection_dict = {}
    for edge in graph.all_edge_list:
        rejection_dict[edge.index] = rs.choice(
            [True, False], 1, p=[edge.p_reject, (1 - edge.p_reject)]
        )[0]
    return rejection_dict


def experiment(args):
    edge_selection_rs = np.random.RandomState(
        seed=args.seed
    )  # for random edge selection
    rs_eval = np.random.RandomState(
        seed=args.seed
    )  # for randomly sampling during objective evaluation
    rs_greedy = np.random.RandomState(seed=args.seed)  # for greedy edge selection
    rs_edge_dist = np.random.RandomState(seed=args.seed)  # for assigning edge probs

    # generate an output file
    name_str = ""
    if args.graph_type != "random":
        if args.input_type == "single":
            name_str = "_" + os.path.basename(args.input_data)

    output_file = generate_filepath(
        args.output_dir, f"{args.expt_type}_expt" + name_str, "csv"
    )

    # generate a file to write results from each fail/accept outcome
    if args.expt_type == "singlestage":
        outcome_file = generate_filepath(
            args.output_dir, f"{args.expt_type}_expt_outcomes" + name_str, "csv"
        )
    else:
        outcome_file = None

    # generate a logfile
    log_file = generate_filepath(
        args.output_dir, f"{args.expt_type}_expt_LOGS" + name_str, "txt"
    )
    formatter = logging.Formatter(LOG_FORMAT)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"beginning {args.expt_type} experiment")
    logger.info("generating output file: {}".format(output_file))
    if args.expt_type == "singlestage":
        logger.info("generating outcomes file: {}".format(outcome_file))

    # list of output column names (singlestage)
    col_list = [
        "graph_name",
        "num_matchable_edges",
        "num_stages",
        "method",
        "edge_budget",
        "num_selected_edges",
        "mean_score",
        "std_score",
        "num_eval_leaf_samples",
        "mcts_level_time_limit",
        "total_solve_time",
        "other_data",
    ]

    col_list_outcomes = [
        "graph_name",
        "num_stages",
        "method",
        "edge_budget",
        "num_selected_edges",
        "outcome_number",
        "outcome_prob",
        "score",
        "deterministic_matching_score",
        "failaware_matching_score",
        "other_data",
    ]

    col_list_multistage = [
        "graph_name",
        "num_queried_edges",
        "edge_budget",
        "method",
        "outcome_number",
        "score",
        "deterministic_matching_score",
        "failaware_matching_score",
        "other_data",
    ]

    if (args.expt_type == "multistage") and args.kpd_data:
        raise NotImplementedError

    if args.kpd_data:
        header_cols = (
            col_list
            + KPDData.col_list(col_prefix="matched")
            + KPDData.col_list(col_prefix="graph")
        )
        header_cols_outcomes = (
            col_list_outcomes
            + KPDData.col_list(col_prefix="matched")
            + KPDData.col_list(col_prefix="graph")
        )
    else:
        if args.expt_type == "singlestage":
            header_cols = col_list
            header_cols_outcomes = col_list_outcomes
        else:
            header_cols = col_list_multistage

    # write the file header: write experiment parameters to the first line of the output file
    with open(output_file, "w") as f:
        f.write(str(args) + "\n")
        f.write((DELIMITER.join(header_cols) + "\n"))

    if args.expt_type == "singlestage":
        # write the header for the outcomes file
        with open(outcome_file, "w") as f:
            f.write(str(args) + "\n")
            f.write((DELIMITER.join(header_cols_outcomes) + "\n"))
        # store all results in a namedtuple
        Result = namedtuple("Result", col_list)
    else:
        Result = namedtuple("Result", col_list_multistage)

    # the end of the result string depends on whether we're saving kpd data
    result_end = ";" if args.kpd_data else "\n"

    def result_to_str(result, end="\n"):
        """return a string representation of a result, for writing to a csv"""
        result_str = (DELIMITER.join(len(col_list) * ["{}"]) + end).format(
            result.graph_name,
            result.num_matchable_edges,
            result.num_stages,
            result.method,
            result.edge_budget,
            result.num_selected_edges,
            result.mean_score,
            result.std_score,
            result.num_eval_leaf_samples,
            result.mcts_level_time_limit,
            result.total_solve_time,
            result.other_data,
        )
        return result_str

    def get_multistage_realization_results(
        outcome_number,
        edge_list,
        edge_rejections,
        graph,
        method_name,
        deterministic_sol,
        fail_aware_sol,
        other_data={},
    ):
        """take a list of queried edges from a multistage experiment (in order they were queried), and a dict of
        pre-determined edge rejections.

        for each edge in the sequence, calculate the intermediate value of querying each edge. compare to the value of

        return a list of strings to be written to the outcomes file

        """
        result_str_list = []

        for edge_budget in range(1, args.max_query_edges + 1):
            # for each subset of edges...
            accepted_edges = []
            rejected_edges = []
            num_queried_edges = min(edge_budget, len(edge_list))
            for e in edge_list[:num_queried_edges]:
                if edge_rejections[e.index]:
                    rejected_edges.append(e)
                else:
                    accepted_edges.append(e)

            total_weight, kpd_data = evaluate_accept_reject_outcome(
                accepted_edges, rejected_edges, graph, use_kpd_data=False
            )

            other_data["edge_list"] = [e.index for e in edge_list[0:num_queried_edges]]
            other_data["reject_list"] = [
                edge_rejections[e.index] for e in edge_list[0:num_queried_edges]
            ]

            # calculate the realized weight of deterministic and failure-aware
            edge_acceptance = [
                not edge_rejections[e.index] for e in edge_list[0:num_queried_edges]
            ]
            det_score = eval_sol_on_outcome(
                edge_list[0:num_queried_edges],
                edge_acceptance,
                deterministic_sol,
                graph,
            )
            fail_aware_score = eval_sol_on_outcome(
                edge_list[0:num_queried_edges], edge_acceptance, fail_aware_sol, graph
            )

            result_str = (
                DELIMITER.join(9 * ["{}"]).format(
                    graph.name,
                    num_queried_edges,  # num stages
                    edge_budget,  # edge budget
                    method_name,
                    outcome_number,
                    total_weight,
                    det_score,
                    fail_aware_score,
                    other_data,
                )
                + "\n"
            )

            result_str_list.append(result_str)

        return result_str_list

    def get_single_stage_realization_result(
        edge_list,
        edge_budget,
        graph,
        method_name,
        solve_time,
        deterministic_sol,
        fail_aware_sol,
        other_data={},
        outcome_file=None,
    ):
        """calculate the value of the edge list"""

        # initialize the graph config with the correct edge success probability
        graph.init_optconfig(args.matching_edge_success_prob)

        if args.kpd_data:
            (
                weight_list,
                outcome_prob_list,
                kpd_data_list,
                edge_acceptance_list,
            ) = evaluate_edge_list_with_data(
                edge_list, graph, args.num_eval_leaf_samples, rs_eval
            )
        else:
            weight_list, outcome_prob_list, edge_acceptance_list = evaluate_edge_list(
                edge_list, graph, args.num_eval_leaf_samples, rs_eval
            )

        other_data["edge_list"] = [e.index for e in edge_list]

        # write each outcome
        if outcome_file is not None:
            # first cols are always the same
            outcome_cols_1 = (
                DELIMITER.join(5 * ["{}"]).format(
                    graph.name,
                    1,  # num stages
                    method_name,
                    edge_budget,
                    len(edge_list),
                )
                + DELIMITER
            )

            if args.kpd_data:
                with open(outcome_file, "a") as f:
                    for i, (weight, prob, kpd_data, edge_acceptance) in enumerate(
                        zip(
                            weight_list,
                            outcome_prob_list,
                            kpd_data_list,
                            edge_acceptance_list,
                        )
                    ):

                        # calculate expected matching weights of deterministic and fail-aware matchings, conditional on
                        # this outcome
                        deterministic_wt = eval_sol_on_outcome(
                            edge_list, edge_acceptance, deterministic_sol, graph
                        )
                        failaware_wt = eval_sol_on_outcome(
                            edge_list, edge_acceptance, fail_aware_sol, graph
                        )

                        other_data_new = dict(other_data)
                        other_data_new["edge_acceptance"] = list(edge_acceptance)
                        outcome_cols_2 = (
                            DELIMITER.join(6 * ["{}"]).format(
                                i,
                                prob,
                                weight,
                                deterministic_wt,
                                failaware_wt,
                                other_data_new,
                            )
                            + DELIMITER
                        )
                        kpd_str = (
                            kpd_data.to_string(delimiter=DELIMITER, end=DELIMITER)
                            + graph_kpd_data_string
                        )
                        f.write(outcome_cols_1 + outcome_cols_2 + kpd_str)
            else:
                with open(outcome_file, "a") as f:
                    for i, (weight, prob, edge_acceptance) in enumerate(
                        zip(weight_list, outcome_prob_list, edge_acceptance_list)
                    ):

                        # calculate expected matching weights of deterministic and fail-aware matchings, conditional on
                        # this outcome
                        deterministic_wt = eval_sol_on_outcome(
                            edge_list, edge_acceptance, deterministic_sol, graph
                        )
                        failaware_wt = eval_sol_on_outcome(
                            edge_list, edge_acceptance, fail_aware_sol, graph
                        )

                        other_data_new = dict(other_data)
                        other_data_new["edge_acceptance"] = edge_acceptance
                        outcome_cols_2 = (
                            DELIMITER.join(6 * ["{}"]).format(
                                i,
                                prob,
                                weight,
                                deterministic_wt,
                                failaware_wt,
                                other_data_new,
                            )
                            + "\n"
                        )
                        f.write(outcome_cols_1 + outcome_cols_2)

        # calculate aggregate results
        mean_weight = np.mean(weight_list)
        std_weight = np.sqrt(np.mean(np.power(weight_list - mean_weight, 2)))
        if args.kpd_data:
            kpd_data = KPDData()
            for prob, x in zip(outcome_prob_list, kpd_data_list):
                kpd_data = kpd_data + x * prob
            kpd_data = kpd_data * (1 / sum(outcome_prob_list))

        result_str = result_to_str(
            Result(
                graph_name=graph.name,
                num_matchable_edges=len(graph.matchable_edge_list),
                num_stages=1,
                method=method_name,
                edge_budget=edge_budget,
                num_selected_edges=len(edge_list),
                mean_score=mean_weight,
                std_score=std_weight,
                num_eval_leaf_samples=args.num_eval_leaf_samples,
                mcts_level_time_limit=args.mcts_level_time_limit,
                total_solve_time=solve_time,
                other_data=other_data,
            ),
            end=result_end,
        )

        if args.kpd_data:
            result_str = (
                result_str
                + kpd_data.to_string(delimiter=DELIMITER, end=DELIMITER)
                + graph_kpd_data_string
            )

        return result_str

    # ------------------------------------
    # Read data
    # ------------------------------------

    if args.input_type == "multiple":
        # create a generator to step through each graph
        if args.graph_type == "cmu":
            graph_generator = get_cmu_graphs(
                args.input_data,
                args.cycle_cap,
                args.chain_cap,
                drop_edge_frac=args.drop_edge_frac,
                seed=args.seed,
            )
        if args.graph_type == "unos":
            graph_generator = get_unos_graphs(
                args.input_data, args.cycle_cap, args.chain_cap,
            )
    if args.input_type == "single":
        if args.graph_type == "cmu":
            graph_generator = [
                get_single_cmu_graph(
                    args.input_data,
                    args.cycle_cap,
                    args.chain_cap,
                    drop_edge_frac=args.drop_edge_frac,
                    seed=args.seed,
                )
            ]
        if args.graph_type == "unos":
            graph_generator = [
                read_unos_graph_with_data(
                    args.input_data, args.cycle_cap, args.chain_cap,
                )
            ]

    if args.graph_type == "random":
        rs_graphs = np.random.RandomState(
            seed=args.seed
        )  # for generating random graphs
        max_graph_seed = 10000
        graph_generator = [
            GraphStructure.er_randomgraph(
                args.n_randomgraph,
                args.p_randomgraph,
                rs_graphs.randint(max_graph_seed),
                cycle_cap=args.cycle_cap,
                chain_cap=args.chain_cap,
            )
            for _ in range(args.num_random_graphs)
        ]

    # set up the initialization function for edge weights
    if args.edge_dist_type == "simple":
        # all edges have p_reject 0.5, p_success_accept = 1.0, and p_success_noquery = 0.5
        initialize_edges = lambda graph: add_uniform_probabilities(graph, 0.5, 1.0, 0.5)
        p_success_failaware = 0.5
    if args.edge_dist_type == "kpd":
        initialize_edges = lambda graph: add_kpd_probabilities(graph, rs_edge_dist)
        p_success_failaware = 0.3135

    # run the experiment for each graph
    for graph in graph_generator:

        # initialize edge dist
        initialize_edges(graph)

        logger.info(f"running {args.expt_type} methods on graph: {graph.name}")

        # check if the graph has more than one matching
        graph.init_optconfig(edge_success_prob=1.0)
        deterministic_sol = solve_picef_model(graph.optconfig)

        logger.info(
            f"number of edges in max-wt matching: {len(deterministic_sol.matching_edges)}"
        )

        if len(deterministic_sol.matching_edges) == 0:
            logger.info("skipping graph: no matchings")
            continue

        # remove the matched edges and check whether a feasible solution exists:
        deterministic_sol_2 = solve_picef_model(
            graph.optconfig, remove_edges=deterministic_sol.matching_edges
        )
        logger.info(
            f"number of edges after removing max-wt matching: {len(deterministic_sol_2.matching_edges)}"
        )

        if len(deterministic_sol_2.matching_edges) == 0:
            logger.info("skipping graph: only one matching")
            continue

        if args.kpd_data:
            graph_kpd_data_string = graph.get_kpd_data().to_string(
                delimiter=DELIMITER, end="\n"
            )

        if args.expt_type == "singlestage":
            # ------------------------------------
            # Single-stage: previous methods
            # ------------------------------------

            # ------------------------------------
            # one-shot matching: failure-aware
            # ------------------------------------
            logger.info("running failure-aware")
            method_name = "failure_aware"
            t0 = time.time()

            # initialize the model with correct edge success probability
            graph.init_optconfig(edge_success_prob=p_success_failaware)
            fail_aware_sol = solve_picef_model(graph.optconfig)

            solve_time = time.time() - t0

            # now calculate expected weight and get kpd data
            expected_weight = expected_matching_weight_noquery(
                fail_aware_sol, graph.graph, graph.altruists
            )
            if args.kpd_data:
                kpd_data = get_matching_kpd_data(fail_aware_sol, graph)

            result_str = result_to_str(
                Result(
                    graph_name=graph.name,
                    num_matchable_edges=len(graph.matchable_edge_list),
                    num_stages=1,
                    method=method_name,
                    edge_budget=0,
                    num_selected_edges=0,
                    mean_score=expected_weight,
                    std_score=NULL_OUTPUT,
                    num_eval_leaf_samples=NULL_OUTPUT,
                    mcts_level_time_limit=NULL_OUTPUT,
                    total_solve_time=solve_time,
                    other_data=NULL_OUTPUT,
                ),
                end=result_end,
            )

            if args.kpd_data:
                result_str = (
                    result_str
                    + kpd_data.to_string(delimiter=DELIMITER, end=DELIMITER)
                    + graph_kpd_data_string
                )

            logger.info("writing {} results".format(method_name))
            with open(output_file, "a") as f:
                f.write(result_str)

            # ------------------------------------
            # one-shot matching: deterministic
            # ------------------------------------
            logger.info("running deterministic")
            # calculate the deterministic matching
            method_name = "deterministic"
            t0 = time.time()
            # initialize the model with correct edge success probability
            graph.init_optconfig(edge_success_prob=1.0)
            deterministic_sol = solve_picef_model(graph.optconfig)
            solve_time = time.time() - t0

            # calculate expected weight
            expected_weight = expected_matching_weight_noquery(
                deterministic_sol, graph.graph, graph.altruists
            )
            if args.kpd_data:
                kpd_data = get_matching_kpd_data(deterministic_sol, graph)

            result_str = result_to_str(
                Result(
                    graph_name=graph.name,
                    num_matchable_edges=len(graph.matchable_edge_list),
                    num_stages=1,
                    method=method_name,
                    edge_budget=0,
                    num_selected_edges=0,
                    mean_score=expected_weight,
                    std_score=NULL_OUTPUT,
                    num_eval_leaf_samples=NULL_OUTPUT,
                    mcts_level_time_limit=NULL_OUTPUT,
                    total_solve_time=solve_time,
                    other_data=NULL_OUTPUT,
                ),
                end=result_end,
            )

            if args.kpd_data:
                result_str = (
                    result_str
                    + kpd_data.to_string(delimiter=DELIMITER, end=DELIMITER)
                    + graph_kpd_data_string
                )

            logger.info("writing {} results".format(method_name))
            with open(output_file, "a") as f:
                f.write(result_str)

            # ------------------------------------
            # single-stage ignorance is almost bliss
            # ------------------------------------
            egde_budget_set = set(range(1, args.max_query_edges + 1))
            if args.num_iiab_rounds > 0:
                logger.info("running iiab")
                t0 = time.time()
                iiab_edge_list = ignorance_is_almost_bliss(
                    graph, args.matching_edge_success_prob, rounds=args.num_iiab_rounds
                )
                solve_time = time.time() - t0

                # get the k values that correspond to each iiab round
                iiab_k_values = [len(edge_list) for edge_list in iiab_edge_list]

                # construct a list of edge budgets that includes these IIAB values
                egde_budget_set.update(iiab_k_values)
                logger.info(
                    f"final edge budget list after adding IIAB edge budget(s): {egde_budget_set}"
                )

                logger.info("gathering IIAB results")
                iiab_results = []
                for r in range(args.num_iiab_rounds):
                    method_name = "iiab_{}".format((r + 1))
                    iiab_results.append(
                        get_single_stage_realization_result(
                            iiab_edge_list[r],
                            None,
                            graph,
                            method_name,
                            solve_time,
                            deterministic_sol,
                            fail_aware_sol,
                            outcome_file=outcome_file,
                        )
                    )

                logger.info("writing iiab results")
                with open(output_file, "a") as f:
                    for result in iiab_results:
                        f.write(result)

            # ------------------------------------
            # Single-stage greedy (use only arg-specified edge budgets)
            # ------------------------------------
            logger.info("running greedy")
            method_name = "greedy"
            graph.init_optconfig(edge_success_prob=args.matching_edge_success_prob)
            t0 = time.time()
            greedy_edges = greedily_choose_edge_list_timelimit(
                graph,
                max(egde_budget_set),
                rs_greedy,
                args.time_limit,
                args.num_leaf_samples,
                args.max_level_for_pruning,
            )
            solve_time = time.time() - t0

            # get the greedy result for each incremental edge subset
            num_greedy_edges = len(greedy_edges)
            for num_query_edges in range(
                1, num_greedy_edges + 1
            ):  # sorted(egde_budget_set):
                result = get_single_stage_realization_result(
                    greedy_edges[0:num_query_edges],
                    num_query_edges,
                    graph,
                    method_name,
                    solve_time,
                    deterministic_sol,
                    fail_aware_sol,
                    outcome_file=outcome_file,
                )

                logger.info(
                    "writing {} results for {} edges".format(
                        method_name, num_query_edges
                    )
                )
                with open(output_file, "a") as f:
                    f.write(result)

            # ------------------------------------
            # Single-stage random (use all edge budgets including IIAB)
            # ------------------------------------
            method_results = []
            logger.info("running random edge selection")
            for num_query_edges in egde_budget_set:
                for i in range(args.num_random_samples):
                    method_name = "random_{}".format(i)
                    solve_time = NULL_OUTPUT
                    e_random = edge_selection_rs.choice(
                        graph.matchable_edge_list,
                        min(num_query_edges, len(graph.matchable_edge_list)),
                        replace=False,
                    )

                    method_results.append(
                        get_single_stage_realization_result(
                            e_random,
                            num_query_edges,
                            graph,
                            method_name,
                            solve_time,
                            deterministic_sol,
                            fail_aware_sol,
                            outcome_file=outcome_file,
                        )
                    )

            logger.info("writing random results")
            with open(output_file, "a") as f:
                for result in method_results:
                    f.write(result)

            # ---- new ---- run MCTS only once

            # ------------------------------------
            # Single-stage: MCTS
            # ------------------------------------

            logger.info(
                "running MCTS single-stage with num_query_edges={}".format(
                    max(egde_budget_set)
                )
            )
            graph.init_optconfig(edge_success_prob=args.matching_edge_success_prob)

            method_name = "mcts_singlestage"
            search_tree = create_edge_subset_search_tree(
                graph,
                max(egde_budget_set),
                num_leaf_samples=args.num_leaf_samples,
                num_simulations=1,
                max_nodes=args.max_nodes,
                max_level_for_pruning=args.max_level_for_pruning,
            )

            start_time = time.time()
            if args.mcts_level_time_limit > 0:
                search_tree.train(
                    level_time_limit=args.mcts_level_time_limit,
                    max_level=args.mcts_max_level,
                )
            runtime = time.time() - start_time

            best_node = EdgeSubsetNode([], search_tree, calc_value_first=True)
            best_num_edges = 0
            for num_edges, node in search_tree.best_node_by_level.items():
                logger.info(f"MCTS solution for {num_edges} edges: {node}")
                if node.value > best_node.value:
                    logger.info("... MCTS returns this node")
                    best_node = node
                    best_num_edges = num_edges
                else:
                    logger.info(
                        f"... MCTS returns solution with {best_num_edges} edges: {best_node}"
                    )

                result = get_single_stage_realization_result(
                    best_node.edge_list,
                    num_edges,  # edge budget
                    graph,
                    method_name,
                    runtime,
                    deterministic_sol,
                    fail_aware_sol,
                    other_data={"num_level_samples": search_tree.num_level_samples},
                    outcome_file=outcome_file,
                )

                # logger.info(f"writing {method_name} results".format(method_name))
                with open(output_file, "a") as f:
                    f.write(result)

        elif args.expt_type == "multistage":
            # ------------------------------------
            # Multi-stage: our method (pre-training)
            # ------------------------------------

            # ------------------------------------
            # Single-stage: previous methods
            # ------------------------------------

            # ------------------------------------
            # one-shot matching: failure-aware
            # ------------------------------------
            logger.info("running failure-aware")
            method_name = "failure_aware"
            t0 = time.time()

            # initialize the model with correct edge success probability
            graph.init_optconfig(edge_success_prob=p_success_failaware)
            fail_aware_sol = solve_picef_model(graph.optconfig)

            solve_time = time.time() - t0

            # now calculate expected weight and get kpd data
            expected_weight = expected_matching_weight_noquery(
                fail_aware_sol, graph.graph, graph.altruists
            )
            if args.kpd_data:
                kpd_data = get_matching_kpd_data(fail_aware_sol, graph)

            result_str = (
                DELIMITER.join(9 * ["{}"]).format(
                    graph.name,
                    0,  # num stages
                    0,  # edge budget
                    method_name,
                    NULL_OUTPUT,
                    expected_weight,
                    NULL_OUTPUT,
                    NULL_OUTPUT,
                    {},
                )
                + "\n"
            )

            logger.info(f"writing {method_name} results")
            with open(output_file, "a") as f:
                f.write(result_str)

            # ------------------------------------
            # one-shot matching: deterministic
            # ------------------------------------
            logger.info("running deterministic")
            # calculate the deterministic matching
            method_name = "deterministic"
            t0 = time.time()
            # initialize the model with correct edge success probability
            graph.init_optconfig(edge_success_prob=1.0)
            deterministic_sol = solve_picef_model(graph.optconfig)

            # calculate expected weight
            expected_weight = expected_matching_weight_noquery(
                deterministic_sol, graph.graph, graph.altruists
            )
            if args.kpd_data:
                kpd_data = get_matching_kpd_data(deterministic_sol, graph)

            result_str = (
                DELIMITER.join(9 * ["{}"]).format(
                    graph.name,
                    0,  # num stages
                    0,  # edge budget
                    method_name,
                    NULL_OUTPUT,
                    expected_weight,
                    NULL_OUTPUT,
                    NULL_OUTPUT,
                    {},
                )
                + "\n"
            )

            logger.info("writing {} results".format(method_name))
            with open(output_file, "a") as f:
                f.write(result_str)

            for trial in range(args.num_multistage_trials):
                #
                logger.info(f"starting multistage trial {trial + 1}")
                # define the pre-selected edge outcomes for the trial
                edge_rejections = sample_edge_rejections(graph, edge_selection_rs)

                # set the multistage tree to train on this trial only

                t0 = time.time()
                _, tree = create_multistage_search_tree(graph, args.max_query_edges, args.stage_time_limit)
                mcts_policy_fn = create_mcts_policy()
                mcts_queried_edges = simulate_policy_multistage(
                    mcts_policy_fn, graph, edge_rejections, args.max_query_edges, tree
                )

                solve_time = time.time() - t0
                method_name = "mcts_multistage"

                result_list = get_multistage_realization_results(
                    trial,
                    mcts_queried_edges,
                    edge_rejections,
                    graph,
                    method_name,
                    deterministic_sol,
                    fail_aware_sol,
                    other_data={"solve_time": solve_time},
                )

                logger.info(f"writing {method_name} results")
                with open(output_file, "a") as f:
                    for result_str in result_list:
                        f.write(result_str)

                # ------------------------------------
                # Multi-stage: greedy
                # ------------------------------------
                logger.info(f"running multi-stage greedy")
                graph.init_optconfig(args.matching_edge_success_prob)
                greedy_policy_fn = create_greedy_policy(args.time_limit)
                t0 = time.time()
                queried_edges = simulate_policy_multistage(
                    greedy_policy_fn, graph, edge_rejections, args.max_query_edges, None
                )

                solve_time = time.time() - t0
                method_name = "greedy_multistage"

                result_list = get_multistage_realization_results(
                    trial,
                    queried_edges,
                    edge_rejections,
                    graph,
                    method_name,
                    deterministic_sol,
                    fail_aware_sol,
                    other_data={"solve_time": solve_time},
                )

                logger.info(f"writing {method_name} results")
                with open(output_file, "a") as f:
                    for result_str in result_list:
                        f.write(result_str)

                # ------------------------------------
                # Multi-stage: random
                # ------------------------------------
                logger.info("running random policy")

                def random_policy_fn(graph, queried_edges, edge_outcomes, tree):
                    remaining_edges = [
                        e for e in graph.matchable_edge_list if e not in queried_edges
                    ]
                    if len(remaining_edges) == 0:
                        return None
                    return edge_selection_rs.choice(remaining_edges, 1)[0]

                for i in range(args.num_random_samples):
                    method_name = f"random_{i}"
                    queried_edges = simulate_policy_multistage(
                        random_policy_fn, graph, edge_rejections, args.max_query_edges, None
                    )

                    result_list = get_multistage_realization_results(
                        trial,
                        queried_edges,
                        edge_rejections,
                        graph,
                        method_name,
                        deterministic_sol,
                        fail_aware_sol,
                        other_data={"solve_time": solve_time},
                    )

                    logger.info(f"writing {method_name} results")
                    with open(output_file, "a") as f:
                        for result_str in result_list:
                            f.write(result_str)

        logger.info("done")


def main():
    # run the experiment. sample usage:
    parser = argparse.ArgumentParser()

    # experiment params
    parser.add_argument(
        "--seed", type=int, help="random seed for experiments", default=0
    )
    parser.add_argument(
        "--expt-type",
        type=str,
        choices=["singlestage", "multistage"],
        help="type of experiment {'singlestage' | 'multistage'}",
    )
    parser.add_argument(
        "--max-query-edges",
        type=int,
        help="max. number of edges to select for pre-screening for each method.",
    )
    parser.add_argument(
        "--num-random-samples",
        type=int,
        default=10,
        help="number of times to evaluate a random edge set",
    )
    parser.add_argument(
        "--num-iiab-rounds",
        type=int,
        default=3,
        help="number of rounds of ignorance is almost bliss to run",
    )
    parser.add_argument(
        "--kpd-data",
        action="store_true",
        default=False,
        help="if true, save and output KPD-related data",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100000,
        help="maximum number of nodes to store in memory during tree search",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=50,
        help="number of simulations to run when sampling non-terminal nodes in MCTS",
    )
    parser.add_argument(
        "--max-level-for-pruning",
        type=int,
        default=2,
        help="passed to edge_selection_tree.get_candidate_edges, for pruning the search tree when possible (used by "
        "greedy and MCTS)",
    )
    parser.add_argument(
        "--edge-dist-type",
        type=str,
        choices=["simple", "kpd"],
        help="type of edge distribution. {'simple' | 'kpd'}",
    )

    # i/o
    parser.add_argument(
        "--input-data",
        type=str,
        help="input directory, containing exchange graph files",
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["unos", "cmu", "random"],
        help="type of graphs {'unos' | 'cmu' | 'random'}",
    )
    # only if graph type is random
    parser.add_argument(
        "--num-random-graphs",
        type=int,
        default=0,
        help="number of random graphs to run experiments on",
    )
    parser.add_argument(
        "--n-randomgraph",
        type=int,
        default=0,
        help="number of vertices for each random graph",
    )
    parser.add_argument(
        "--p-randomgraph",
        type=float,
        default=0.0,
        help="probability of edges in Erdos-Renyi random graph",
    )

    parser.add_argument(
        "--input-type",
        type=str,
        choices=["single", "multiple"],
        help="set to 'single' if --input-data provides a single exchange, set to 'multiple' if --input-data provides a "
        "directory of exchanges.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="output directory, where an output csv will be written",
    )

    # matching params
    parser.add_argument("--chain-cap", type=int, default=4, help="chain cap")
    parser.add_argument("--cycle-cap", type=int, default=3, help="cycle cap")
    parser.add_argument(
        "--num-leaf-samples",
        type=int,
        default=50,
        help="number of samples per graph evaluation during edge search",
    )
    parser.add_argument(
        "--matching-edge-success-prob",
        type=float,
        default=1.0,
        help="edge success probability used by the final matching policy (and IIAB)",
    )
    parser.add_argument(
        "--num-eval-leaf-samples",
        type=int,
        default=50,
        help="number of samples for evaluating each edge set. if -1, sample entire distribution.",
    )
    parser.add_argument(
        "--num-multistage-trials",
        type=int,
        default=30,
        help="number of times to run the multistage experiments with different edge outcomes",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="if set, use a fixed arg string for debugging. otherwise, parse args.",
        default=False,
    )

    parser.add_argument(
        "--time-limit", type=int, default=-1, help="max runtime for greedy, in seconds",
    )

    # mcts-specific args
    parser.add_argument(
        "--mcts-level-time-limit",
        type=int,
        default=60,
        help="max runtime for each level of MCTS",
    )
    parser.add_argument(
        "--mcts-max-level",
        type=int,
        default=15,
        help="max level of exploration for MCTS",
    )

    parser.add_argument(
        "--stage-time-limit",
        type=int,
        default=20.0,
        help="max runtime allowed for each non-root stage for mcts and greedy",
    )
    parser.add_argument(
        "--drop-edge-frac",
        type=float,
        default=0.0,
        help="fraction of edges to drop in each random sample (only for CMU-type graphs)",
    )

    args = parser.parse_args()

    validate_args(args)
    experiment(args)


def validate_args(args):
    # validate args
    assert args.max_query_edges is not None
    assert args.expt_type is not None
    assert args.edge_dist_type is not None
    assert args.graph_type is not None
    assert args.output_dir is not None
    if args.graph_type == "random":
        assert args.num_random_graphs > 0
        assert args.p_randomgraph > 0
        assert args.p_randomgraph < 1
        assert args.n_randomgraph > 0
        if args.edge_dist_type == "kpd":
            raise NotImplementedError
    else:
        assert args.input_data is not None
        assert args.input_type is not None


if __name__ == "__main__":
    main()

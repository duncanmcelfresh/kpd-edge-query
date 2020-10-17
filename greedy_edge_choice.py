import time
import numpy as np

from edge_selection_tree import (
    evaluate_edge_list,
    evaluate_accept_reject_outcome,
    get_candidate_edges,
)

from kidney_ip import solve_picef_model
from multistage_edge_selection import get_candidate_edges_fixed_rejections
from utils import succeeded_failed_edges, get_logger

logger = get_logger()


def greedily_choose_edge_list_timelimit(
    graph,
    num_edges,
    edge_selection_rs,
    time_limit,
    num_leaf_samples,
    max_level_for_pruning,
):
    edge_list = []

    start_time = time.time()

    for i in range(num_edges):
        max_val = 0.0
        best_edge = None

        # get the set of candidate edges to add to edge_list (the first edge to add will be in this list)
        candidate_edges = get_candidate_edges(edge_list, graph, max_level_for_pruning)

        if len(candidate_edges) == 0:
            logger.info(f"greedy has no candidate edges at step {i + 1}")
            break

        for e in candidate_edges:

            # if we're out of time
            if (time.time() - start_time) > time_limit:

                # randomly add all remaining edges to sample set
                logger.info(f"greedy ran out of time after {len(edge_list)} edges")
                remaining_edge_count = num_edges - len(edge_list)

                for j in range(remaining_edge_count):
                    if len(candidate_edges) == 0:
                        logger.info(
                            f"greedy has no candidate edges at step {len(edge_list) + 1}"
                        )
                        break
                    new_edge = edge_selection_rs.choice(list(candidate_edges))
                    edge_list.append(new_edge)
                    candidate_edges = get_candidate_edges(
                        edge_list, graph, max_level_for_pruning
                    )

                return edge_list

            else:

                weight_list, prob_list, _ = evaluate_edge_list(
                    edge_list + [e], graph, num_leaf_samples, edge_selection_rs
                )
                edge_value = np.mean(weight_list)

                if edge_value > max_val:
                    best_edge = e
                    max_val = edge_value

        candidate_edges.remove(best_edge)
        edge_list.append(best_edge)
    return edge_list


def create_greedy_policy(time_limit):
    def greedily_get_next_edge_timelimit(graph, queried_edges, edge_rejections, tree):
        best_edge = None

        rejected_edges = []
        accepted_edges = []
        for e, rejected in zip(queried_edges, edge_rejections):
            if rejected:
                rejected_edges.append(e)
            else:
                accepted_edges.append(e)

        start_time = time.time()
        candidate_edges = get_candidate_edges_fixed_rejections(
            queried_edges, edge_rejections, graph
        )

        if len(candidate_edges) == 0:
            logger.info(
                f"multistage greedy ran out of edges on stage {len(queried_edges)}"
            )
            return None

        best_edge_value = -1
        for e in candidate_edges:
            if (time.time() - start_time) > time_limit:
                logger.info(
                    f"multistage greedy ran out of time on stage {len(queried_edges)}"
                )
                if best_edge is None:
                    return np.random.choice(graph.candidate_edges)
                else:
                    return best_edge

            e_success_value, _ = evaluate_accept_reject_outcome(
                accepted_edges + [e], rejected_edges, graph
            )
            e_fail_value, _ = evaluate_accept_reject_outcome(
                accepted_edges, rejected_edges + [e], graph
            )
            edge_value = (
                1 - e.p_reject
            ) * e_success_value + e.p_reject * e_fail_value
            if edge_value > best_edge_value:
                best_edge = e
                best_edge_value = edge_value

        return best_edge

    return greedily_get_next_edge_timelimit


def greedily_get_next_edge(graph, queried_edges, edge_outcomes):

    best_edge_value = -1
    best_edge = None
    succeeded_list, failed_list = succeeded_failed_edges(queried_edges, edge_outcomes)
    for e in graph.all_edge_list:
        if e not in queried_edges:
            e_success_value, _ = evaluate_accept_reject_outcome(
                succeeded_list + [e], failed_list, graph
            )
            e_fail_value, _ = evaluate_accept_reject_outcome(
                succeeded_list, failed_list + [e], graph
            )
            edge_value = (1 - e.p_reject) * e_success_value + e.p_reject * e_fail_value
            if edge_value > best_edge_value:
                best_edge = e
                best_edge_value = edge_value
    return best_edge


def ignorance_is_almost_bliss(graph, edge_success_prob, rounds=1):
    """select all of the edges for the fail-aware matching with specified edge_success_prob"""
    edge_list_stages = []
    total_edge_list = []

    # re-initialize the graph's optconfig and model object with edge success prob
    graph.init_optconfig(edge_success_prob=edge_success_prob)

    for r in range(rounds):
        new_edges = []
        opt_result = solve_picef_model(graph.optconfig, remove_edges=total_edge_list)

        for cycle in opt_result.cycle_obj:
            new_edges.extend(cycle.edges)

        # calculate expected chain weight
        for chain in opt_result.chains:
            chain_edges = chain.get_edge_objs(graph.graph, graph.altruists)
            new_edges.extend(chain_edges)

        # add the new edges to the total edge list
        total_edge_list.extend(new_edges)

        # total_edge_list is the new query list
        edge_list_stages.append(total_edge_list)

    # re-initialize the graph's optconfig and model object with edge success prob 1 (default)
    graph.init_optconfig(edge_success_prob=1.0)

    return edge_list_stages

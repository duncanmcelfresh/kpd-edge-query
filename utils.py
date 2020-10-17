import hashlib
import json
import logging
import os
import re
import time

import numpy as np

from kpd_data import KPDData

LOG_FORMAT = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"


def get_logger(logfile=None):
    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logger


def stable_hash(x):
    """hash a general python object by hashing its json representation. unlike hash() this is reproducible"""
    hash_str = hashlib.md5(json.dumps(x).encode("utf-8")).hexdigest()
    return hash_str, int(hash_str, 16)


def edge_list_hash(edge_list):
    """return the stable hash of a list of edges. this hash is reproducible, and does not depend on edge order"""
    return stable_hash(sorted([e.hash_int for e in edge_list]))


def query_node_hash(accepted_edges, rejected_edges, new_edge):
    return stable_hash(
        (
            sorted([e.hash_int for e in accepted_edges]),
            sorted([e.hash_int for e in rejected_edges]),
            new_edge.hash_int,
        )
    )

def outcome_node_hash(accepted_edges, rejected_edges):
    return stable_hash(
        (
            sorted([e.hash_int for e in accepted_edges]),
            sorted([e.hash_int for e in rejected_edges]),
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
# kidney exchange helper functions
# ----------------------------------------------------------------------------------------------------------------------

def initialize_random_edge_weights(graph, rs_greedy):
    add_uniform_probabilities(graph, 0.5, 1.0, 0.5)
    for e in graph.all_edge_list:
        e.weight = 100.0 + rs_greedy.uniform(1,10)

    graph.init_optconfig()

def add_kpd_probabilities(graph, rs):
    """
    input: graphstructure object

    for each edge in graph.all_edge_list, set the following:

    e.p_reject: U[0.25, 0.43] (according to UNOS estimates, mean = .34)

    e.p_success_accept
    - if e.sensitized: U[0.2, 0.5] (mean = .35)
    - if not e.sensitized: U[0.9, 1.0] (mean = 0.95)

    e.p_success_noquery
    - if e.sensitized: U[0.0, 0.2] (mean = 0.1)
    - if not e.sensitized: U[0.8, 0.9] (mean = 0.85)

    the overall mean non-queried failure prob is roughly: 0.34 + (1 - 0.34) * 0.525 = 0.6865
    overall mean success prob. is 0.3135
    (assuming equal proportion sensitized and non-sensitized)
    """
    for e in graph.all_edge_list:
        e.p_reject = rs.uniform(0.25, 0.43)
        if graph.lookup_recip_sensitized(e.tgt.aux_id) == "Y":
            e.p_success_accept = rs.uniform(0.2, 0.5)
            e.p_success_noquery = rs.uniform(0.0, 0.2)
        else:
            e.p_success_accept = rs.uniform(0.9, 1.0)
            e.p_success_noquery = rs.uniform(0.8, 0.9)


def add_uniform_probabilities(graph, p_reject, p_success_accept, p_success_noquery):
    for e in graph.all_edge_list:
        e.p_reject = p_reject
        e.p_success_accept = p_success_accept
        e.p_success_noquery = p_success_noquery


def succeeded_failed_edges(edge_outcomes, queried_edges):
    succeeded_edges = []
    failed_edges = []
    for e, o in zip(queried_edges, edge_outcomes):
        if o:
            succeeded_edges.append(e)
        else:
            failed_edges.append(e)
    return succeeded_edges, failed_edges


def expected_matching_weight_noquery(sol, digraph, ndds):
    """calculated expected weight of a kidney exchange solution, assuming no edges are queried"""
    expected_weight = 0.0

    # calculate expected cycle weight
    for cycle in sol.cycle_obj:
        total_wt = 0.0
        total_prob = 1.0
        for e in cycle.edges:
            total_wt += e.weight
            total_prob *= e.p_success_noquery

        expected_weight += total_wt * total_prob

    # calculate expected chain weight
    for chain in sol.chains:
        chain_success_prob = 1.0
        chain_edges = chain.get_edge_objs(digraph, ndds)
        chain_weight = 0.0
        for e in chain_edges:
            chain_success_prob *= e.p_success_noquery
            chain_weight += e.weight * chain_success_prob
        expected_weight += chain_weight

    return expected_weight


def get_matching_kpd_data(opt_solution, graph):
    """
    take a KEX solution, consisting of a list of cycles and a list of chains. return a populated KPDData object
    """

    kpd_data = KPDData()

    # keep track of matched vertices
    matched_vs = []
    # calculate expected cycle weight
    for cycle in opt_solution.cycle_obj:

        # increment abo counts for all donors/recips in the cycle
        kpd_data.cycle_counts[len(cycle.edges)] += 1
        for e in cycle.edges:
            kpd_data.donor_abo[graph.lookup_donor_abo(e.data["donor_id"])] += 1
            kpd_data.recip_abo[graph.lookup_recip_abo(e.tgt.aux_id)] += 1
            matched_vs.append(e.tgt)

            # increment the (recip, donor) abo for the pair at the source of the edge
            recip_abo = graph.lookup_recip_abo(e.src.aux_id)
            donor_abo = graph.lookup_donor_abo(e.data["donor_id"])
            kpd_data.pair_abo[(recip_abo, donor_abo)] += 1

    # calculate expected chain weight
    matched_ndds = []
    for chain in opt_solution.chains:
        chain_edges = chain.get_edge_objs(graph.graph, graph.altruists)
        kpd_data.chain_counts[len(chain_edges)] += 1

        ndd_donor = graph.altruists[chain_edges[0].src_id]
        matched_ndds.append(ndd_donor)
        ndd_donor_abo = graph.lookup_donor_abo(ndd_donor.aux_id)
        kpd_data.ndd_abo[ndd_donor_abo] += 1

        for e in chain_edges[1:]:
            matched_vs.append(e.src)
            # increment abo counts for donor and recip, and donor/recip pair of the source edge
            src_recip_abo = graph.lookup_recip_abo(e.src.aux_id)
            src_donor_abo = graph.lookup_donor_abo(e.data["donor_id"])
            kpd_data.recip_abo[src_recip_abo] += 1
            kpd_data.donor_abo[src_donor_abo] += 1
            kpd_data.pair_abo[(src_recip_abo, src_donor_abo)] += 1
        # increment abo counts for the final recipient in the chain
        # (we ignore the final donors/pairs because they are not used)
        kpd_data.recip_abo[graph.lookup_recip_abo(chain_edges[-1].tgt.aux_id)] += 1

    # get bin counts for in/out degree
    in_deg_list = [v.in_degree for v in matched_vs]
    out_deg_list = [v.out_degree for v in matched_vs]
    ndd_out_deg_list = [ndd.out_degree for ndd in matched_ndds]
    kpd_data.in_deg_counts = np.histogram(in_deg_list, bins=KPDData.in_deg_bin_edges)[0]
    kpd_data.out_deg_counts = np.histogram(
        out_deg_list, bins=KPDData.out_deg_bin_edges
    )[0]
    kpd_data.ndd_out_deg_counts = np.histogram(
        ndd_out_deg_list, bins=KPDData.out_deg_bin_edges
    )[0]

    highly_sensitized_list = [
        graph.lookup_recip_sensitized(v.aux_id) for v in matched_vs
    ]

    kpd_data.high_low_sensitized_count = [
        sum(1 for sens in highly_sensitized_list if sens == "Y"),
        sum(1 for sens in highly_sensitized_list if sens == "N"),
    ]

    return kpd_data


# ----------------------------------------------------------------------------------------------------------------------
# non-kidney-exchange helper functions
# ----------------------------------------------------------------------------------------------------------------------


def simple_string(complex_string, non_numeric=False):
    if non_numeric:
        return re.sub(r"[^A-Za-z]+", "", complex_string).lower()
    else:
        return re.sub(r"[^A-Za-z0-9]+", "", complex_string).lower()


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + "_%s." + extension) % timestr
    return os.path.join(output_dir, output_string)

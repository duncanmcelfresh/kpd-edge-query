# implementation of the PICEF formulation, adapted from https://github.com/jamestrimble/kidney_solver

from gurobipy import *

import kidney_utils
from gurobi_functions import optimize, create_mip_model
from kidney_digraph import Cycle, failure_aware_cycle_weight, cycle_weight

###################################################################################################
#                                                                                                 #
#                                  Code used by all formulations                                  #
#                                                                                                 #
###################################################################################################


class OptConfig(object):
    """The inputs (problem instance and parameters) for an optimisation run

    Data members:
        digraph
        ndds
        max_cycle
        max_chain
        verbose: True if and only if Gurobi output should be written to screen and log file
        timelimit
        edge_success_prob
    """

    def __init__(
        self,
        digraph,
        ndds,
        max_cycle,
        max_chain,
        verbose=False,
        timelimit=None,
        edge_success_prob=1,
        name=None,
        use_chains=True,
    ):
        self.digraph = digraph
        self.ndds = ndds
        self.max_cycle = max_cycle
        self.max_chain = max_chain
        self.verbose = verbose
        self.timelimit = timelimit
        self.edge_success_prob = edge_success_prob
        self.edge_failure_prob = 1.0 - self.edge_success_prob
        self.name = name
        self.use_chains = use_chains

        # fields to be populated by create_picef_model()
        self.m = None
        self.cycles = None
        self.cycle_vars = None
        self.cycle_list = None


class OptSolution(object):
    """An optimal solution for a kidney-exchange problem instance.

    Data members:
        ip_model: The Gurobi Model object
        cycles: A list of cycles in the optimal solution, each represented
            as a list of vertices
        chains: A list of chains in the optimal solution, each represented
            as a Chain object
        total_weight: The total weight of the solution
    """

    def __init__(
        self,
        ip_model,
        cycles,
        chains,
        digraph,
        edge_success_prob=1,
        infeasible=False,
        robust_weight=0,
        optimistic_weight=0,
        cycle_obj=None,
        cycle_cap=None,
        chain_cap=None,
        matching_edges=None,
        alpha_var=None,
    ):
        self.ip_model = ip_model
        self.cycles = cycles
        self.chains = chains
        self.digraph = digraph
        self.infeasible = infeasible
        if self.infeasible:
            self.total_weight = 0
        else:
            self.total_weight = sum(c.weight for c in chains) + sum(
                failure_aware_cycle_weight(c, digraph, edge_success_prob)
                for c in cycles
            )
        self.edge_success_prob = edge_success_prob
        self.cycle_obj = cycle_obj
        self.matching_edges = matching_edges
        self.robust_weight = robust_weight
        self.optimistic_weight = optimistic_weight
        self.cycle_cap = cycle_cap
        self.chain_cap = chain_cap
        self.alpha_var = alpha_var

        if ip_model is not None:
            self.timeout = ip_model.status == GRB.TIME_LIMIT
        else:
            self.timeout = False

    def same_matching_edges(self, other):
        if len(self.matching_edges) != len(other.matching_edges):
            return False
        for self_e in self.matching_edges:
            edge_found = False
            for other_e in other.matching_edges:
                if (self_e.src_id == other_e.src_id) and (
                    self_e.tgt.id == other_e.tgt.id
                ):
                    edge_found = True
                    break
            if not edge_found:
                return False
        return True

    def add_matching_edges(self, ndds):
        """Set attribute 'matching_edges' using self.cycle_obj, self.chains, and self.digraph"""

        matching_edges = []

        for ch in self.chains:
            chain_edges = []
            tgt_id = ch.vtx_indices[0]
            for e in ndds[ch.ndd_index].edges:
                if e.tgt.id == tgt_id:
                    chain_edges.append(e)
            if len(chain_edges) == 0:
                raise Exception("NDD edge not found")
            for i in range(len(ch.vtx_indices) - 1):
                chain_edges.append(
                    self.digraph.adj_mat[ch.vtx_indices[i]][ch.vtx_indices[i + 1]]
                )
            if len(chain_edges) != (len(ch.vtx_indices)):
                raise Exception(
                    "Chain contains %d edges, but only %d edges found"
                    % (len(ch.vtx_indices), len(chain_edges))
                )
            matching_edges.extend(chain_edges)

        for cy in self.cycle_obj:
            cycle_edges = []
            for i in range(len(cy.vs) - 1):
                cycle_edges.append(self.digraph.adj_mat[cy.vs[i].id][cy.vs[i + 1].id])
            # add final edge
            cycle_edges.append(self.digraph.adj_mat[cy.vs[-1].id][cy.vs[0].id])
            if len(cycle_edges) != len(cy.vs):
                raise Exception(
                    "Cycle contains %d vertices, but only %d edges found"
                    % (len(cy.vs), len(cycle_edges))
                )
            matching_edges.extend(cycle_edges)

        self.matching_edges = matching_edges


###################################################################################################
#                                                                                                 #
#                  Chain vars and constraints (used by HPIEF', HPIEF'' and PICEF)                 #
#                                                                                                 #
###################################################################################################


def add_chain_vars_and_constraints(
    digraph,
    ndds,
    use_chains,
    max_chain,
    m,
    vtx_to_vars,
    store_edge_positions=False,
    check_edge_success=False,
):
    """Add the IP variables and constraints for chains in PICEF and HPIEF'.

    Args:
        ndds: a list of NDDs in the instance
        use_chains: boolean: True if chains should be used
        max_chain: the chain cap
        m: The Gurobi model
        vtx_to_vars: A list such that for each Vertex v in the Digraph,
            vtx_to_vars[v.id] will contain the Gurobi variables representing
            edges pointing to v.
        store_edge_positions: if this is True, then an attribute grb_var_positions
            will be added to edges that have associated Gurobi variables.
            edge.grb_var_positions[i] will indicate the position of the edge respresented
            by edge.grb_vars[i]. (default: False)
    """

    if use_chains:  # max_chain > 0:
        for v in digraph.vs:
            v.grb_vars_in = [[] for i in range(max_chain - 1)]
            v.grb_vars_out = [[] for i in range(max_chain - 1)]
            v.edges_in = [[] for i in range(max_chain - 1)]
            v.edges_out = [[] for i in range(max_chain - 1)]

        for ndd in ndds:
            ndd_edge_vars = []
            for e in ndd.edges:
                edge_var = m.addVar(vtype=GRB.BINARY)
                if check_edge_success:
                    if not e.success:
                        m.addConstr(edge_var == 0)
                e.edge_var = edge_var
                e.used_var = edge_var
                ndd_edge_vars.append(edge_var)

                vtx_to_vars[e.tgt.id].append(edge_var)
                if max_chain > 1:
                    e.tgt.grb_vars_in[0].append(edge_var)
                    e.tgt.edges_in[0].append(e)

            m.update()
            m.addConstr(quicksum(ndd_edge_vars) <= 1)

        dists_from_ndd = kidney_utils.get_dist_from_nearest_ndd(digraph, ndds)

        # Add pair->pair edge variables, indexed by position in chain
        # e.grb_var are the chain variables for each edge.
        for e in digraph.es:
            e.grb_vars = []

            if store_edge_positions:
                e.grb_var_positions = []
            for i in range(max_chain - 1):
                if dists_from_ndd[e.src.id] <= i + 1:
                    edge_var = m.addVar(vtype=GRB.BINARY)
                    if check_edge_success:
                        if not e.success:
                            m.addConstr(edge_var == 0)
                    e.grb_vars.append(edge_var)

                    if store_edge_positions:
                        e.grb_var_positions.append(i + 1)
                    vtx_to_vars[e.tgt.id].append(edge_var)
                    e.src.grb_vars_out[i].append(edge_var)
                    e.src.edges_out[i].append(e)

                    if i < max_chain - 2:
                        e.tgt.grb_vars_in[i + 1].append(edge_var)
                        e.tgt.edges_in[i + 1].append(e)

        # At each chain position, sum of edges into a vertex must be >= sum of edges out
        for i in range(max_chain - 1):
            for v in digraph.vs:
                m.addConstr(quicksum(v.grb_vars_in[i]) >= quicksum(v.grb_vars_out[i]))

        m.update()


###################################################################################################
#                                                                                                 #
#                                              PICEF                                              #
#                                                                                                 #
###################################################################################################


def create_picef_model(cfg, check_edge_success=False):
    """Optimise using the PICEF formulation.

    Args:
        cfg: an OptConfig object
        check_edge_success: (bool). if True, check if each edge has e.success = False. if e.success=False, the edge cannot
            be used.

    Returns:
        an OptSolution object
    """

    cycles = cfg.digraph.find_cycles(cfg.max_cycle)

    m = create_mip_model(time_lim=cfg.timelimit, verbose=cfg.verbose)
    m.params.method = -1

    cycle_vars = [m.addVar(vtype=GRB.BINARY) for __ in cycles]

    vtx_to_vars = [[] for __ in cfg.digraph.vs]

    add_chain_vars_and_constraints(
        cfg.digraph,
        cfg.ndds,
        cfg.use_chains,
        cfg.max_chain,
        m,
        vtx_to_vars,
        store_edge_positions=True,
        check_edge_success=check_edge_success,
    )

    for i, c in enumerate(cycles):
        for v in c:
            vtx_to_vars[v.id].append(cycle_vars[i])

    for l in vtx_to_vars:
        if len(l) > 0:
            m.addConstr(quicksum(l) <= 1)

    # add variables for each pair-pair edge indicating whether it is used in a cycle or chain
    for e in cfg.digraph.es:
        used_in_cycle = []
        for var, c in zip(cycle_vars, cycles):
            if kidney_utils.cycle_contains_edge(c, e):
                used_in_cycle.append(var)

        used_var = m.addVar(vtype=GRB.INTEGER)
        if check_edge_success:
            if not e.success:
                m.addConstr(used_var == 0)

        if cfg.use_chains:
            m.addConstr(used_var == quicksum(used_in_cycle) + quicksum(e.grb_vars))
        else:
            m.addConstr(used_var == quicksum(used_in_cycle))
        e.used_var = used_var

    # add cycle objects
    cycle_list = []
    for c, var in zip(cycles, cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.weight = failure_aware_cycle_weight(
            c_obj.vs, cfg.digraph, cfg.edge_success_prob
        )
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    # add objective
    if not cfg.use_chains:
        obj_expr = quicksum(
            failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob) * var
            for c, var in zip(cycles, cycle_vars)
        )
    elif cfg.edge_success_prob == 1:
        obj_expr = (
            quicksum(
                cycle_weight(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)
            )
            + quicksum(e.weight * e.edge_var for ndd in cfg.ndds for e in ndd.edges)
            + quicksum(e.weight * var for e in cfg.digraph.es for var in e.grb_vars)
        )
    else:
        obj_expr = (
            quicksum(
                failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob) * var
                for c, var in zip(cycles, cycle_vars)
            )
            + quicksum(
                e.weight * cfg.edge_success_prob * e.edge_var
                for ndd in cfg.ndds
                for e in ndd.edges
            )
            + quicksum(
                e.weight * cfg.edge_success_prob ** (pos + 1) * var
                for e in cfg.digraph.es
                for var, pos in zip(e.grb_vars, e.grb_var_positions)
            )
        )
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    m.update()

    # attach the necessary objects to the optconfig
    cfg.m = m
    cfg.cycles = cycles
    cfg.cycle_vars = cycle_vars
    cfg.cycle_list = cycle_list


def optimize_picef(cfg, check_edge_success=False):
    """create and solve a picef model, and return the solution"""

    if cfg.m is None:
        create_picef_model(cfg, check_edge_success=check_edge_success)

    optimize(cfg.m)

    if cfg.use_chains:
        matching_chains = kidney_utils.get_optimal_chains(
            cfg.digraph, cfg.ndds, cfg.edge_success_prob
        )
    else:
        matching_chains = []

    cycles_used = [c for c, v in zip(cfg.cycles, cfg.cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cfg.cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(
        ip_model=cfg.m,
        cycles=cycles_used,
        cycle_obj=cycle_obj,
        chains=matching_chains,
        digraph=cfg.digraph,
        edge_success_prob=cfg.edge_success_prob,
        cycle_cap=cfg.max_chain,
        chain_cap=cfg.max_cycle,
    )
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(
        sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain
    )
    return cycle_obj, matching_chains, sol


def solve_picef_model(cfg, remove_edges=[]):
    """
    solve a picef model using a config object, and return the solution

    if remove_edges is provided, disallow these edges from being used.
    """

    for e in remove_edges:
        e.used_var.setAttr(GRB.Attr.UB, 0.0)

    optimize(cfg.m)

    if cfg.use_chains:
        matching_chains = kidney_utils.get_optimal_chains(
            cfg.digraph, cfg.ndds, cfg.edge_success_prob
        )
    else:
        matching_chains = []

    cycles_used = [c for c, v in zip(cfg.cycles, cfg.cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cfg.cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(
        ip_model=cfg.m,
        cycles=cycles_used,
        cycle_obj=cycle_obj,
        chains=matching_chains,
        digraph=cfg.digraph,
        edge_success_prob=cfg.edge_success_prob,
        cycle_cap=cfg.max_chain,
        chain_cap=cfg.max_cycle,
    )
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(
        sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain
    )

    # allow removed edges to be used again
    for e in remove_edges:
        e.used_var.setAttr(GRB.Attr.UB, 1.0)

    return sol

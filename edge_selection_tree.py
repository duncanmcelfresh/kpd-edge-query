# graph classes for single-stage edge selection
import time

import numpy as np
from gurobipy import *
import scipy
from scipy import special

from kidney_ip import solve_picef_model
from utils import get_matching_kpd_data, edge_list_hash, get_logger

logger = get_logger()

LARGE_UCB_CONST = 1000

# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------


def create_edge_subset_search_tree(
    graph,
    edge_budget,
    num_leaf_samples=-1,
    num_simulations=10,
    max_nodes=100000,
    max_level_for_pruning=2,
):
    """initialize a search tree and root node for the single-stage edge selection problem"""
    search_tree = SearchTree(
        graph,
        edge_budget,
        num_leaf_samples=num_leaf_samples,
        num_simulations=num_simulations,
        max_nodes=max_nodes,
        max_level_for_pruning=max_level_for_pruning,
    )
    root = EdgeSubsetNode([], search_tree, calc_value_first=True)
    search_tree.add_root(root)
    return search_tree


def evaluate_edge_list(edge_list, graph, num_leaf_samples, rs):
    """
    evaluate the expected outcome of this edge query list. if num_leaf_samples is large enough to completely estimate
    the mean, then call evaluate_edge_list_exact (i.e., without sampling). otherwise, evaluate by sampling.

    inputs:
    - edges (iterable(edge objects)) : list of edges to simulate realizations after querying
    - graph: GraphStructure object
    - num_leaf_samples: number of outcomes to sample
    """

    if num_leaf_samples >= 2 ** len(edge_list):
        return evaluate_edge_list_exact(edge_list, graph)

    weight_list = np.zeros(num_leaf_samples)
    prob_list = np.zeros(num_leaf_samples)
    edge_acceptance_list = []
    for i in range(num_leaf_samples):
        u_rvs = rs.uniform(0, 1, len(edge_list))
        p_reject_list = np.array([e.p_reject for e in edge_list])
        edge_acceptance = (
            u_rvs >= p_reject_list
        )  # if the uniform RV is greater than p_reject, then accept

        prob_list[i], weight_list[i] = eval_graph_on_outcome(
            edge_list, edge_acceptance, graph
        )
        edge_acceptance_list.append(edge_acceptance)

    return weight_list, prob_list, edge_acceptance_list


def evaluate_edge_list_with_data(edge_list, graph, num_leaf_samples, rs):
    """
    identical to evaluate_edge_list(), but return kpd-related data
    """

    if num_leaf_samples >= 2 ** len(edge_list):
        return evaluate_edge_list_exact_with_data(edge_list, graph)

    weight_list = np.zeros(num_leaf_samples)
    kpd_data_list = [None for _ in range(num_leaf_samples)]
    prob_list = np.zeros(num_leaf_samples)
    edge_acceptance_list = []
    for i in range(num_leaf_samples):
        u_rvs = rs.uniform(0, 1, len(edge_list))
        p_reject_list = np.array([e.p_reject for e in edge_list])
        edge_acceptance = (
            u_rvs >= p_reject_list
        )  # if the uniform RV is greater than p_reject, then accept

        (
            prob_list[i],
            weight_list[i],
            kpd_data_list[i],
        ) = eval_graph_on_outcome_with_data(edge_list, graph, edge_acceptance)
        edge_acceptance_list.append(edge_acceptance)

    return weight_list, prob_list, kpd_data_list, edge_acceptance_list


def evaluate_edge_list_exact(edge_list, graph):
    """
    exactly evaluate the mean and stddev of the outcome of this edge query list

    inputs:
    - edge_list (list(edge objects)) : list of edges to simulate realizations after querying
    - graph: GraphStructure object

    find the expected quality of this edge list, by enumerating all realizations of the queried edges (edge_list). for
    each realization:
    - calculate the probability of this realization
    - find the max-weight deterministic matching in this outcome (not using rejected edges)
    - calculate the expected weight in this outcome
    """

    # iterate through all reject/accpet outcomes for the selected edges
    # outcome[i] = True means that edge i is accepted, rejected if false
    num_outcomes = 2 ** len(edge_list)
    weight_list = np.zeros(num_outcomes)
    prob_list = np.zeros(num_outcomes)
    edge_acceptance_list = []
    for i, edge_acceptance in enumerate(
        itertools.product([True, False], repeat=len(edge_list))
    ):  # for every possible outcome

        prob_list[i], weight_list[i] = eval_graph_on_outcome(
            edge_list, edge_acceptance, graph
        )
        edge_acceptance_list.append(edge_acceptance)

    return weight_list, prob_list, edge_acceptance_list


def evaluate_edge_list_exact_with_data(edge_list, graph):
    """
    identical to evaluate_edge_list_exact(), but return kpd-related data
    """

    # iterate through all reject/accpet outcomes for the selected edges
    # outcome[i] = True means that edge i is accepted, rejected if false
    num_outcomes = 2 ** len(edge_list)
    weight_list = np.zeros(num_outcomes)
    prob_list = np.zeros(num_outcomes)
    kpd_data_list = [None for _ in range(num_outcomes)]
    edge_acceptance_list = []
    for i, edge_acceptance in enumerate(
        itertools.product([True, False], repeat=len(edge_list))
    ):  # for every possible outcome
        (
            prob_list[i],
            weight_list[i],
            kpd_data_list[i],
        ) = eval_graph_on_outcome_with_data(edge_list, graph, edge_acceptance)
        edge_acceptance_list.append(edge_acceptance)

    return weight_list, prob_list, kpd_data_list, edge_acceptance_list


def eval_graph_on_outcome(edge_list, edge_acceptance, graph):
    """
    evaluate the expected weight of a max-weight matching on a graph, given the outcome for a set of edges

    args:
    - edges (list(edge)): set of edges which are queried
    - edge_acceptance (list(bool)): if edge_acceptance[i] = True, then edges[i] is accepted, otherwise it is rejected.
    - graph (GrpahStructure): kidney exchange graph - must have an OptConfig object as a property
    """

    opt_solution = solve_picef_model(
        graph.optconfig,
        remove_edges=[e for e, accept in zip(edge_list, edge_acceptance) if not accept],
    )

    # print(f"matching edges: {[str(e) for e in opt_solution.matching_edges]}")
    # print(f"matching score: {sum(e.weight for e in opt_solution.matching_edges)}")

    # calculate the probability of this outcome occurring
    outcome_prob = np.prod(
        [
            (1.0 - e.p_reject) if accept else e.p_reject
            for e, accept in zip(edge_list, edge_acceptance)
        ]
    )

    # calculate total expected weight
    total_weight = 0.0

    # expected cycle weight
    for cycle in opt_solution.cycle_obj:
        cycle_weight = sum(e.weight for e in cycle.edges)
        cycle_success_prob = np.prod(
            [
                e.p_success_accept
                if e in edge_list
                else e.p_success_noquery  # if edges are matched and in edge_list, then they succeeded
                for e in cycle.edges
            ]
        )

        total_weight += cycle_weight * cycle_success_prob

    # expected chain weight
    for chain in opt_solution.chains:
        chain_success_prob = 1.0
        chain_edges = chain.get_edge_objs(graph.graph, graph.altruists)
        chain_weight = 0.0
        for e in chain_edges:
            chain_success_prob *= (
                e.p_success_accept if e in edge_list else e.p_success_noquery
            )
            chain_weight += e.weight * chain_success_prob

        total_weight += chain_weight

    return outcome_prob, total_weight


def eval_sol_on_outcome(edge_list, edge_acceptance, sol, graph):
    """
    evaluate the expected weight of a solution, given the query outcome for a set of edges

    similar to eval_graph_on_outcome(), only take a solution (matching) rather than a graph. the main difference is that
    the solution provided may use rejected edges, which reduces the final matching weight.

    the sol should be found using the graph object

    args:
    - edges (list(edge)): set of edges which are queried
    - edge_acceptance (list(bool)): if edge_acceptance[i] = True, then edges[i] is accepted, otherwise it is rejected.
    - sol (OptSol): kidney exchange matching
    - graph (GrpahStructure): kidney exchange graph - must have an OptConfig object as a property
    """

    # calculate total expected weight
    total_weight = 0.0

    reject_edges = [e for e, a in zip(edge_list, edge_acceptance) if not a]
    accept_edges = [e for e, a in zip(edge_list, edge_acceptance) if a]

    # expected cycle weight
    for cycle in sol.cycle_obj:
        # skip this cycle if any edges failed
        if not any(e in reject_edges for e in cycle.edges):
            cycle_weight = sum(e.weight for e in cycle.edges)
            cycle_success_prob = np.prod(
                [
                    e.p_success_accept
                    if e in accept_edges
                    else e.p_success_noquery  # if edges are matched and in edge_list, then they succeeded
                    for e in cycle.edges
                ]
            )
            total_weight += cycle_weight * cycle_success_prob

    # expected chain weight
    for chain in sol.chains:
        chain_success_prob = 1.0
        chain_edges = chain.get_edge_objs(graph.graph, graph.altruists)
        chain_weight = 0.0
        for e in chain_edges:
            if e in reject_edges:
                break
            chain_success_prob *= (
                e.p_success_accept if e in edge_list else e.p_success_noquery
            )
            chain_weight += e.weight * chain_success_prob

        total_weight += chain_weight

    return total_weight


def eval_graph_on_outcome_with_data(edge_list, graph, edge_acceptance):
    """
    identical to eval_graph_on_outcome(), but returns kpd-related data
    """

    # solve the deterministic picef model, with realized edge failures
    opt_solution = solve_picef_model(
        graph.optconfig,
        remove_edges=[e for e, accept in zip(edge_list, edge_acceptance) if not accept],
    )

    # calculate the probability of this outcome occurring
    outcome_prob = np.prod(
        [
            (1.0 - e.p_reject) if accept else e.p_reject
            for e, accept in zip(edge_list, edge_acceptance)
        ]
    )

    # get kpd-related data on this solution
    kpd_data = get_matching_kpd_data(opt_solution, graph)

    # calculate total expected weight
    total_weight = 0.0

    # calculate expected cycle weight
    for cycle in opt_solution.cycle_obj:
        cycle_weight = sum(e.weight for e in cycle.edges)
        cycle_success_prob = np.prod(
            [
                e.p_success_accept if e in edge_list else e.p_success_noquery
                for e in cycle.edges
            ]
        )

        total_weight += cycle_weight * cycle_success_prob

    # calculate expected chain weight
    for chain in opt_solution.chains:
        chain_success_prob = 1.0
        chain_edges = chain.get_edge_objs(graph.graph, graph.altruists)
        chain_weight = 0.0
        for i, e in enumerate(chain_edges):
            chain_success_prob *= (
                e.p_success_accept if e in edge_list else e.p_success_noquery
            )
            chain_weight += e.weight * chain_success_prob

        total_weight += chain_weight

    return outcome_prob, total_weight, kpd_data


def evaluate_accept_reject_outcome(
    accepted_edges, rejected_edges, graph, use_kpd_data=False
):
    combined_edge_list = accepted_edges + rejected_edges
    edge_acceptance = [True] * len(accepted_edges) + [False] * len(rejected_edges)

    if not use_kpd_data:
        outcome_prob, total_weight = eval_graph_on_outcome(
            combined_edge_list, edge_acceptance, graph
        )
        kpd_data = None
    else:
        outcome_prob, total_weight, kpd_data = eval_graph_on_outcome_with_data(
            combined_edge_list, graph, edge_acceptance
        )
    return total_weight, kpd_data


def get_candidate_edges(edge_list, graph, max_level_for_pruning):
    """
    get the set of potential edges to query, given that edge_list is already in the query set.

    if len(edge_list) <= max_level_for_pruning, then enumerate all fail/succeed outcomes for all edges in edge_list, and
    find the matched edges in each outcome - then return the union of these matched edges.

    otherwise, just return graph.matchable_edges
    """
    if len(edge_list) <= max_level_for_pruning:
        candidate_edges = set()
        for edge_fail_list in itertools.product([True, False], repeat=len(edge_list)):
            opt_solution = solve_picef_model(
                graph.optconfig,
                remove_edges=[
                    e for e, fail in zip(edge_list, edge_fail_list) if not fail
                ],
            )

            # update the set of potential child edges
            candidate_edges.update(opt_solution.matching_edges)
    else:
        # otherwise create a child for each matchable edge
        candidate_edges = set(graph.matchable_edge_list)

    return candidate_edges.difference(edge_list)


# ----------------------------------------------------------------------------------------------------------------------
# classes
# ----------------------------------------------------------------------------------------------------------------------


class SearchTree(object):
    """
    a search tree for selecting an edge subset

    properties:
    - root (object): root node of the tree
    - graph (GraphStructure): kidney exchange graph
    - edge_budget (int): max number of edges to be selected
    - edge_subset_nodes: a dict of all EdgeSubsetNode objects (non-leaf nodes). keys are the hash representation.
        this list is pre-allocated at the beginning of the search, to avoid memory issues.
    - leaf_node_memoize (dict): a dict containing all evaluated leaf nodes
    - num_leaf_samples (int): if >0, evaluate the objval of a leaf node with (num_leaf_samples) random samples. if -1,
        calculate the exact objval.
    - max_level_for_pruning (int): passed to get_candidate_edges
    - max_nodes (int): maximum number of nodes to hold in memory. this determines how many levels of lookahead we can
        use. this number of lookahead steps is determined separately for each graph.
    """

    def __init__(
        self,
        graph,
        edge_budget,
        num_leaf_samples=100,
        max_level_for_pruning=4,
        max_nodes=10000,
        num_simulations=100,
        seed=0,
    ):
        self.root = None
        self.graph = graph
        self.edge_budget = edge_budget
        self.best_node = None
        self.worst_node = None
        self.leaf_node_memoize = {}
        self.edge_subset_nodes = {}
        self.num_level_samples = []
        self.nodes_by_level = {i: {} for i in range(1, self.edge_budget + 1)}
        self.num_times_memoized = 0
        self.num_simulations = num_simulations
        self.num_leaf_samples = num_leaf_samples
        self.max_level_for_pruning = max_level_for_pruning
        self.max_nodes = max_nodes
        self.max_explore_level = self.edge_budget
        self.rs = np.random.RandomState(seed)

        # determine how many levels of lookahead we can use. this is at least 1
        num_edges = len(self.graph.matchable_edge_list)
        assert self.max_nodes >= num_edges

        num_nodes = num_edges
        num_lookahead = 1
        # the total number of nodes in layer l is (|E| choose l)
        for level in range(2, self.edge_budget + 1):
            num_nodes += scipy.special.binom(num_edges, level)
            if num_nodes > self.max_nodes:
                break
            num_lookahead = level

        self.num_lookahead = num_lookahead

        # initialize best nodes by level
        dummy_node = EdgeSubsetNode([], None)
        dummy_node.value = -1.0
        self.best_node_by_level = {
            i: dummy_node for i in range(1, self.edge_budget + 1)
        }

    def add_root(self, root):
        self.root = root
        self.best_node = root
        self.worst_node = root

    def num_nodes(self):
        return sum(len(v) for v in self.nodes_by_level.values())

    def sample(self):
        self.root._sample()

    def add_node(self, edge_list):
        """
        check if the node corresponding to edge_list already exists in the tree. if so, do nothing, otherwise create
        the node and add it to the tree. return the node's hash
        """
        _, node_hash = edge_list_hash(edge_list)
        if node_hash not in self.nodes_by_level[len(edge_list)]:
            self.nodes_by_level[len(edge_list)][node_hash] = EdgeSubsetNode(
                edge_list, self
            )
        return node_hash

    def check_temp_node(self, edge_list):
        """
        create a temporary node without adding it to the search tree, and evaluate the node's value. if this node beats
        the incumbent, replace the incumbent. return the node's value
        """
        new_node = EdgeSubsetNode(edge_list, self, calc_value_first=True)

        if new_node.value > self.best_node.value:
            self.best_node = new_node

        if new_node.value > self.best_node_by_level[new_node.level].value:
            self.best_node_by_level[new_node.level] = new_node

        if new_node.value < self.worst_node.value:
            self.worst_node = new_node

        return new_node.value

    def train(
        self, level_time_limit=600, max_level=100, max_levels_without_improvement=5
    ):
        """
        train the search tree using an iterative process. start with root = {} (empty edge subset) and use the following
        process to step through the tree:
        1) Set L <- current level of the tree (num edges in the root node)
        2) Remove all nodes from self.edge_subset_nodes at level L, if any exist.
        3) Add all nodes in levels at depth L + 1, ..., min{self.edge_budget, L + min(self.num_lookahead)} unless they
            already exist (don't want to overwrite their values)
        4) Run MCTS for a fixed time limit (level_time_limit, in seconds)
            - (see EdgeSubsetNode for details)
        5) If child nodes have exactly K edges, or we're at max_level, terminate, otherwise
            set root <-- highest-UCB child, and return to (1)

        if at the end of exploring a level, the best node for that level is equal to the best node above it by
        max_levels_without_improvement, then stop training. (i.e., if the past <max_levels_without_improvement> levels
        have not helped learn anything about this level, then we haven't learned anything)
        """

        # must start with no edge subset nodes
        assert all(
            len(self.nodes_by_level[i]) == 0 for i in range(1, self.edge_budget + 1)
        )

        self.num_level_samples = []

        for level in range(self.edge_budget):
            if level > max_level:
                logger.info("reached max level, returning incumbent")
                break

            logger.info(f"starting MCTS from level {level}. root: {self.root}")

            # remove all nodes at the current or lower level. this removes the root from nodes_by_level, but root is
            # still accessible from the reference self.root.
            for i in range(1, level + 1):
                self.nodes_by_level[i].clear()

            # if the root has no children, we're done
            self.root.check_create_children()
            if self.root.children == []:
                logger.info(
                    f"root has no children. returning best node: {self.best_node}"
                )
                break

            # run MCTS on the next self.num_lookahead layers. at max_explore_level, evaluate the node directly. for
            # level < max_explore_level, simulate.
            self.max_explore_level = min(level + self.num_lookahead, self.edge_budget)

            num_samples = 0
            start_time = time.time()
            while (time.time() - start_time) < level_time_limit:
                # explore from the current root
                self.root._sample()
                num_samples += 1

            self.num_level_samples.append(num_samples)

            # select the best child and make this the new root
            best_child_hash = self.root.get_ucb_child()
            self.root = self.root.get_child(best_child_hash)

            # if the new root hasn't been visited yet, visit it
            if self.root.num_visits == 0:
                weight_list, prob_list, _ = evaluate_edge_list(
                    self.root.edge_list, self.graph, self.num_leaf_samples, self.rs,
                )
                self.root.value = np.mean(weight_list)
                self.root.num_visits = 1

            logger.info(
                f"finished MCTS from level {level} with {num_samples} samples. incumbent: {self.best_node}"
            )

            # if this level is worst than the last (max_levels_without_improvement) levels, stop
            if level > max_levels_without_improvement:
                stop = True
                for i in range(1, max_levels_without_improvement + 1):
                    if (
                        self.best_node_by_level[
                            level - max_levels_without_improvement
                        ].value
                        < self.best_node_by_level[
                            level - max_levels_without_improvement + i
                        ].value
                    ):
                        stop = False
                        break
                if stop:
                    logger.info(
                        f"the level incumbent for the past {max_levels_without_improvement} levels are worse than the "
                        f"incumbent at level {level - max_levels_without_improvement}. stopping MCTS"
                    )
                    break

        return self.best_node


class EdgeSubsetNode(object):
    """
    A single node of the search tree, representing a feasible set of queried edges
    """

    def __init__(
        self,
        edge_list,
        search_tree,
        explore_with_symmetry_breaking=False,
        calc_value_first=False,
    ):
        self.edge_list = edge_list
        self.total_value = 0
        self.children = None
        self.search_tree = search_tree
        self.root = False
        self.ub = np.inf
        self.num_visits = 0
        self.hash_str, self.hash_int = edge_list_hash(self.edge_list)
        self.value = None
        self.child_edges = None

        if calc_value_first:
            weight_list, prob_list, _ = evaluate_edge_list(
                self.edge_list,
                self.search_tree.graph,
                self.search_tree.num_leaf_samples,
                self.search_tree.rs,
            )
            self.value = np.mean(weight_list)
            self.num_visits = 1

        # if true, explore children in a way that preserves symmetry breaking. this will result in a biased search
        self.explore_with_symmetry_breaking = explore_with_symmetry_breaking

    def __str__(self):
        return f"Edge subset node: {[str(e) for e in self.edge_list]}; value = {self.value}"

    def __eq__(self, other):
        return self.hash_int == other.hash_int

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.hash_int

    @property
    def level(self):
        return len(self.edge_list)

    def _sample(self):
        """
        sample from this node. there are three options:
         - if this node has no children / is a leaf node (level = search_tree.edge_budget), then return the value of
            this node
        - if this node is at level search_tree.max_explore_level and this node is *not* a leaf node, then simulate
        - if this node is not at level search_tree.max_explore_level, then call _sample on a child

        the first time _sample is called, the value of the node is calculated
        """

        if self.value is None:
            # calculate the value of this node
            weight_list, prob_list, _ = evaluate_edge_list(
                self.edge_list,
                self.search_tree.graph,
                self.search_tree.num_leaf_samples,
                self.search_tree.rs,
            )
            self.value = np.mean(weight_list)
            self.num_visits += 1

            # if this node is better than the incumbent, replace the incumbent
            if self.value > self.search_tree.best_node.value:
                self.search_tree.best_node = self

            if self.value > self.search_tree.best_node_by_level[self.level].value:
                self.search_tree.best_node_by_level[self.level] = self

            if self.value < self.search_tree.worst_node.value:
                self.search_tree.worst_node = self

        assert self.level <= self.search_tree.max_explore_level

        # if level < search_tree.max_explore_level, then explore from here
        if self.level < self.search_tree.max_explore_level:
            self.check_create_children()

            if self.children == []:
                # if there are no edges that can be added to this subset, return the node's value
                self.num_visits += 1
                return [self.value]

            # select next child to explore, and return its value
            c = self.get_ucb_child()
            v_new = self.get_child(c)._sample()
            self.total_value += sum(v_new)
            self.num_visits += len(v_new)
            return v_new

        # if level = search_tree.max_explore_level and level < max_level, then simulate
        if (
            self.level == self.search_tree.max_explore_level
            and self.level < self.search_tree.edge_budget
        ):

            if self.child_edges is None:
                self.child_edges = get_candidate_edges(
                    self.edge_list,
                    self.search_tree.graph,
                    self.search_tree.max_level_for_pruning,
                )

            v_new = []
            for _ in range(self.search_tree.num_simulations):
                v_new.extend(self.simulate())
            self.total_value += sum(v_new)
            self.num_visits += len(v_new)
            return v_new

        # if level = search_tree.max_explore_level and level = max_level, then this is a leaf node
        # return the node's value
        if (
            self.level == self.search_tree.max_explore_level
            and self.level == self.search_tree.edge_budget
        ):
            self.num_visits += 1
            return [self.value]

    def simulate(self, max_lookahead=10):
        """
        run a simulation through the search tree starting from this node. only "look ahead" up to max_lookahead levels.
        smaller values of max_lookahead biases the search toward lower levels
        """

        # select the next edge to simulate from
        new_edge_list = [self.search_tree.rs.choice(list(self.child_edges))]

        # get the value of adding an edge to this edge list
        v_list = [self.search_tree.check_temp_node(self.edge_list + new_edge_list)]

        for _ in range(
            min(self.search_tree.edge_budget - self.level - 1, max_lookahead - 1)
        ):
            next_edges = get_candidate_edges(
                self.edge_list + new_edge_list,
                self.search_tree.graph,
                self.search_tree.max_level_for_pruning,
            )
            if len(next_edges) == 0:
                return v_list

            new_edge_list.append(self.search_tree.rs.choice(list(next_edges)))
            v_list.append(
                self.search_tree.check_temp_node(self.edge_list + new_edge_list)
            )

        return v_list

    def get_child(self, child_hash):
        """return the EdgeSubsetNode object corresponding to one of this node's children"""
        return self.search_tree.nodes_by_level[self.level + 1][child_hash]

    def get_random_child(self):
        """return a random child, by selecting a child hash using the search tree's rs"""
        if self.children is None:
            return None
        if self.children == []:
            return None

        return self.get_child(self.search_tree.rs.choice(self.children, 1)[0])

    def check_create_children(self):
        """
        create children if they haven't yet been created.

        generate one child node for each edge that can be added to the edge subset. children are stored as a list of
        hashes, in the form of integers
        """

        if self.children is None:
            # get the set of candidate edges
            if self.child_edges is None:
                self.child_edges = get_candidate_edges(
                    self.edge_list,
                    self.search_tree.graph,
                    self.search_tree.max_level_for_pruning,
                )

            # create children for all potential child edges that aren't already in the edge list
            self.children = []
            for edge in self.child_edges:
                child_hash = self.search_tree.add_node(self.edge_list + [edge])
                self.children.append(child_hash)

    def get_ucb_child(self):

        ucb_values = [
            (c, self.get_child(c).calc_ucb(self.num_visits)) for c in self.children
        ]

        # return any child node with maximal ucb
        _, best_value = max(ucb_values, key=lambda x: x[1])

        best_child = self.search_tree.rs.choice(
            [c for c, v in ucb_values if v == best_value], 1
        )[0]

        return best_child

    def calc_ucb(self, n_visits_above):

        assert n_visits_above > 0

        if (self.num_visits > 0) and (
            self.search_tree.best_node.value != self.search_tree.worst_node.value
        ):
            mean = self.total_value / self.num_visits
            a = (mean - self.search_tree.worst_node.value) / (
                self.search_tree.best_node.value - self.search_tree.worst_node.value
            )
            return a + np.sqrt(np.log(n_visits_above) / self.num_visits)
        else:
            return LARGE_UCB_CONST

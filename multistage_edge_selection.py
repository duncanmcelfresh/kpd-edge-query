import numpy as np
import scipy.special
import time

from edge_selection_tree import evaluate_accept_reject_outcome
from kidney_ip import solve_picef_model
from utils import get_logger, query_node_hash, outcome_node_hash

logger = get_logger()

LARGE_UCB_CONST = 1000


def create_mcts_policy():
    def mcts_get_next_edge(graph, queried_edges, edge_rejections, tree):

        # prepare the tree by resetting the root and removing unneeded query nodes.
        accepted_edges = [e for e, r in zip(queried_edges, edge_rejections) if not r]
        rejected_edges = [e for e, r in zip(queried_edges, edge_rejections) if r]

        new_root = MultistageOutcomeNode(None, accepted_edges, rejected_edges, tree)
        tree.prepare_tree(new_root)

        best_child = tree.search_next_child()

        if best_child is None:
            return None

        return best_child.query_edge

    return mcts_get_next_edge


def create_multistage_search_tree(graph, edge_budget, stage_time_limit):
    search_tree = MultistageSearchTree(
        None, graph, edge_budget, stage_time_limit=stage_time_limit
    )
    root = MultistageOutcomeNode(None, [], [], search_tree)
    search_tree.root = root
    return root, search_tree


def get_candidate_edges_fixed_rejections(edge_list, edge_rejections, graph):
    """
    identical to get_candidate_edges(), but only for a fixed reject/accept outcome

    edge_rejections[i] should be True if edge_list[i] is rejected, and False if accepted
    """
    opt_solution = solve_picef_model(
        graph.optconfig,
        remove_edges=[e for i, e in enumerate(edge_list) if edge_rejections[i]],
    )

    return set(opt_solution.matching_edges).difference(edge_list)


class MultistageSearchTree(object):
    """
    a search tree for the multistage problem. the root node should be a MultistageOutcomeNode object
    """

    def __init__(
        self, root, graph, edge_budget, max_nodes=100000, seed=0, stage_time_limit=60
    ):
        self.root = root
        self.graph = graph
        self.edge_budget = edge_budget
        self.query_nodes = {}
        self.best_bound = 0
        self.worst_bound = np.inf
        self.max_nodes = max_nodes
        self.rs = np.random.RandomState(seed)
        self.stage_time_limit = stage_time_limit

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

    def add_query_node(self, parent, edge):
        """
        check if the query node corresponding to child of the MultistageOutcomeNode parent, with additional query edge
        (edge) exists. if not, add it.return the node's hash
        """
        _, node_hash = query_node_hash(
            parent.accepted_edges, parent.rejected_edges, edge
        )
        if node_hash not in self.query_nodes:
            self.query_nodes[node_hash] = MultistageQueryNode(
                parent, edge, parent.search_tree
            )
        return node_hash

    def search_next_child(self):
        """
        train the search tree from the current root, in order to select the best edge to query next
        """
        logger.info(f"beginning multi-stage MCTS search from node: {self.root}")
        # run MCTS from the root (a MultistageOutcomeNode object), on the next self.num_lookahead layers.
        # at max_explore_level, evaluate the node directly. for level < max_explore_level, simulate.
        self.max_explore_level = min(
            self.root.level + self.num_lookahead, self.edge_budget
        )

        num_samples = 0
        start_time = time.time()
        while (time.time() - start_time) < self.stage_time_limit:
            # explore from the current root
            sample_result = self.root.sample_queries()
            if sample_result is None:
                break
            num_samples += 1

        best_child = self.root.get_best_child()
        if best_child is None:
            logger.info(f"finished multi-stage MCTS search with {num_samples} samples. found no chilren")
            return None

        logger.info(
            f"finished multi-stage MCTS search with {num_samples} samples. best query edge: {best_child.query_edge}"
        )
        return best_child

    def prepare_tree(self, new_root):
        """
        reset the tree's root to a new outcome node. remove all nodes in self.query_nodes
        """
        self.root = new_root
        self.query_nodes.clear()


class MultistageOutcomeNode(object):
    """A node representing an outcome. Its children consist of possible queries to make."""

    def __init__(self, parent, accepted_edges, rejected_edges, search_tree):
        self.accepted_edges = accepted_edges
        self.rejected_edges = rejected_edges
        self.parent = parent
        self.children = None
        self.child_edges = None
        self.search_tree = search_tree
        self._outcome_value = None
        self.level = len(self.accepted_edges) + len(self.rejected_edges)

        self.num_visits = 0

    def __str__(self):
        return f"MultistageOutcomeNode: accepted_edges:{self.accepted_edges}, rejected_edges:{self.rejected_edges}"

    def get_outcome_value(self):
        if self._outcome_value is None:
            self._outcome_value, _ = evaluate_accept_reject_outcome(
                self.accepted_edges,
                self.rejected_edges,
                self.search_tree.graph,
                use_kpd_data=False,
            )

        return self._outcome_value

    def sample_queries(self):
        self.num_visits += 1

        assert self.level <= self.search_tree.max_explore_level

        # this is not a leaf node, and we don't need to simulate
        if self.level < self.search_tree.max_explore_level:
            self.check_create_children()

            if len(self.children) == 0:
                return self.get_outcome_value()

            # select next child to explore
            c = self.get_ucb_child()
            v_new = c.sample_outcomes()

        # if level = search_tree.max_explore_level and level < max_level, then simulate
        if (
            self.level == self.search_tree.max_explore_level
            and self.level < self.search_tree.edge_budget
        ):
            v_new = self.simulate()

        # if this is a leaf node, return its value
        if (
            self.level == self.search_tree.max_explore_level
            and self.level == self.search_tree.edge_budget
        ):
            v_new = self.get_outcome_value()

        if v_new < self.search_tree.worst_bound:
            self.search_tree.worst_bound = v_new
        if v_new > self.search_tree.best_bound:
            self.search_tree.best_bound = v_new

        return v_new

    def check_create_children(self):
        if self.children is None:
            # get the set of candidate edges
            edge_rejections = len(self.accepted_edges) * [False] + len(
                self.rejected_edges
            ) * [True]
            self.child_edges = get_candidate_edges_fixed_rejections(
                self.accepted_edges + self.rejected_edges,
                edge_rejections,
                self.search_tree.graph,
            )

            # create children (EdgeQueryNode objects) for all potential child edges that aren't already in the edge list
            self.children = []
            for edge in self.child_edges:
                child_hash = self.search_tree.add_query_node(self, edge)
                self.children.append(child_hash)

    def get_best_child(self):
        """
        Returns the best child (exploit, no UCB).
        """
        self.check_create_children()

        if len(self.children) > 0:
            mean_values = [
                (c, self.search_tree.query_nodes[c].calc_mean()) for c in self.children
            ]

            _, best_value = max(mean_values, key=lambda x: x[1])

            best_child = self.search_tree.rs.choice(
                [c for c, v in mean_values if v == best_value], 1
            )[0]
            return self.search_tree.query_nodes[best_child]
        else:
            return None

    def get_ucb_child(self):
        ucb_values = [
            (c, self.search_tree.query_nodes[c].calc_ucb(self.num_visits))
            for c in self.children
        ]

        # return any child node with maximal ucb
        _, best_value = max(ucb_values, key=lambda x: x[1])

        best_child = self.search_tree.rs.choice(
            [c for c, v in ucb_values if v == best_value], 1
        )[0]

        return self.search_tree.query_nodes[best_child]

    def simulate(self):
        """randomly select edges and outcomes to complete the search, and return the value"""
        if self.child_edges is None:
            edge_rejections = len(self.accepted_edges) * [False] + len(
                self.rejected_edges
            ) * [True]
            self.child_edges = get_candidate_edges_fixed_rejections(
                self.accepted_edges + self.rejected_edges,
                edge_rejections,
                self.search_tree.graph,
            )

        new_edges = self.search_tree.rs.choice(
            list(self.child_edges),
            min(self.search_tree.edge_budget - self.level, len(self.child_edges)),
        )
        edge_rejections = [self.search_tree.rs.random() < e.p_reject for e in new_edges]

        return evaluate_accept_reject_outcome(
            self.accepted_edges + [e for e, r in zip(new_edges, edge_rejections) if not r],
            self.rejected_edges + [e for e, r in zip(new_edges, edge_rejections) if r],
            self.search_tree.graph,
            use_kpd_data=False,
        )[0]


class MultistageQueryNode(object):
    """A node representing its query. Should always have a parent and its children consist of outcomes."""

    def __init__(self, parent, edge_query, search_tree):
        self.parent = parent
        self.accepted_edges = parent.accepted_edges
        self.rejected_edges = parent.rejected_edges
        self.children = None
        self.child_outcome_probs = None
        self.query_edge = edge_query  # the edge object being queried.
        self.search_tree = search_tree

        self.num_visits = 0
        self.total_value = 0.0

    def sample_outcomes(self):
        # randomly choose a child with some probability
        self.check_create_children()
        self.num_visits += 1
        random_next_outcome = self.search_tree.rs.choice(
            self.children, p=self.child_outcome_probs
        )
        resulting_val = random_next_outcome.sample_queries()
        self.total_value += resulting_val
        return resulting_val

    def check_create_children(self):
        if self.children is None:
            self.children = []
            self.child_outcome_probs = []
            self.children.append(
                MultistageOutcomeNode(
                    self,
                    self.accepted_edges + [self.query_edge],
                    self.rejected_edges,
                    self.search_tree,
                )
            )

            self.children.append(
                MultistageOutcomeNode(
                    self,
                    self.accepted_edges,
                    self.rejected_edges + [self.query_edge],
                    self.search_tree,
                )
            )

            self.child_outcome_probs = [
                1.0 - self.query_edge.p_reject,
                self.query_edge.p_reject,
            ]

    def calc_mean(self):
        if self.num_visits == 0:
            return 0.0
        return self.total_value / self.num_visits

    def calc_ucb(self, n_visits_above):
        if self.num_visits > 0:
            mean = self.total_value / self.num_visits
            if (
                not (
                    np.isclose(
                        self.search_tree.best_bound, self.search_tree.worst_bound
                    )
                )
                and (self.search_tree.best_bound != 0)
                and (self.search_tree.worst_bound != np.inf)
            ):
                a = (mean - self.search_tree.worst_bound) / (
                    self.search_tree.best_bound - self.search_tree.worst_bound
                )
            else:
                a = mean

            return a + np.sqrt(np.log(n_visits_above) / self.num_visits)
        else:
            return (
                LARGE_UCB_CONST - self.num_visits
            )  # kluge to go to unvisited nodes first

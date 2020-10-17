"""A Digraph class which can be used for representing donor-patient pairs
(as vertices) and their compatibilities (as weighted edges), along with
some related methods.
Adapted from https://github.com/jamestrimble/kidney_solver
"""
from collections import deque
import json
from utils import stable_hash


class KidneyReadException(Exception):
    pass


def cycle_weight(cycle, digraph):
    """Calculate the sum of a cycle's edge weights.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
        digraph: The digraph in which this cycle appears.
    """

    return sum(
        digraph.adj_mat[cycle[i - 1].id][cycle[i].id].weight for i in range(len(cycle))
    )


def cycle_weight_weighted(cycle, digraph):
    """Calculate the sum of a cycle's edge weights.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
        digraph: The digraph in which this cycle appears.
    """

    return sum(
        digraph.adj_mat[cycle[i - 1].id][cycle[i].id].weight
        * (1.0 + digraph.adj_mat[cycle[i - 1].id][cycle[i].id].alpha)
        for i in range(len(cycle))
    )


def failure_aware_cycle_weight(cycle, digraph, edge_success_prob):
    """Calculate a cycle's total weight, with edge failures and no backarc recourse.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
            UPDATE: cycle = a Cycle object
        digraph: The digraph in which this cycle appears.
        edge_success_prob: The problem that any given edge will NOT fail
    """

    return sum(
        digraph.adj_mat[cycle[i - 1].id][cycle[i].id].weight for i in range(len(cycle))
    ) * edge_success_prob ** len(cycle)


def failure_aware_cycle_weight_weighted(cycle, digraph, edge_success_prob):
    """Calculate a cycle's total weight, with edge failures and no backarc recourse.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
            UPDATE: cycle = a Cycle object
        digraph: The digraph in which this cycle appears.
        edge_success_prob: The problem that any given edge will NOT fail
    """
    return sum(
        digraph.adj_mat[cycle[i - 1].id][cycle[i].id].weight
        * (1.0 + digraph.adj_mat[cycle[i - 1].id][cycle[i].id].alpha)
        for i in range(len(cycle))
    ) * edge_success_prob ** len(cycle)


def reg_failure_aware_cycle_weight(cycle, digraph, edge_success_prob, theta):
    """Calculate a cycle's total weight, with edge failures and no backarc recourse.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
            UPDATE: cycle = a Cycle object
        digraph: The digraph in which this cycle appears.
        edge_success_prob: The problem that any given edge will NOT fail
    """

    return (
        sum(
            digraph.adj_mat[cycle[i - 1].id][cycle[i].id].weight
            for i in range(len(cycle))
        )
        - theta * len(cycle)
    ) * edge_success_prob ** len(cycle)


class Vertex:
    """A vertex in a directed graph (see the Digraph class)."""

    def __init__(self, id, aux_id=None):
        self.id = id
        self.aux_id = aux_id
        self.donor_set = set()
        self.edges = []
        self.in_degree = 0
        self.out_degree = 0
        self.sensitized = False

    def __str__(self):
        return "V{}".format(self.id)


class Cycle:
    """A cycle in a directed graph.

    Contains:
    - list of vertices, in order
    - list of edges
    - cycle weight
    """

    def __init__(self, vs):
        self.vs = vs
        self.weight = 0
        self.length = len(vs)
        self.discount_frac = 0
        self.edges = []

    def to_dict(self):
        cy_dict = {
            "vs": [v.id for v in self.vs],
            "discount_frac": self.discount_frac,
            "weight": self.weight,
        }
        return cy_dict

    def to_dict_dynamic(self):
        """
        for dynamic experiments, save different info
        """
        cy_dict = {"vs": [v.to_dict_dynamic() for v in self.vs], "weight": self.weight}
        return cy_dict

    @classmethod
    def from_dict(cls, cy_dict, digraph):
        cy = cls([digraph.vs[vi] for vi in cy_dict["vs"]])
        cy.discount_frac = cy_dict["discount_frac"]
        cy.weight = cy_dict["weight"]
        cy.add_edges(digraph.es)
        return cy

    def __cmp__(self, other):
        if min(self.vs) < min(other.vs):
            return -1
        elif min(self.vs) > min(other.vs):
            return 1
        else:
            return 1

    def __len__(self):
        return self.length

    def display(self):
        vtx_str = " ".join(str(v) for v in self.vs)
        return "L=%2d; %15s; weight = %f; discount_frac = %f" % (
            self.length,
            vtx_str,
            self.weight,
            self.discount_frac,
        )

    def contains_edge(self, e):
        if e.src in self.vs:
            i = self.vs.index(e.src)
            if e.tgt == self.vs[(i + 1) % self.length]:
                return True
            else:
                return False
        return False

    def add_edges(self, es):
        # create an unordered list of edges in the cycle
        self.edges = [e for e in es if self.contains_edge(e)]


class Edge:
    """An edge in a directed graph (see the Digraph class)."""

    def __init__(self, id, weight, src, tgt, discount=0, fail=False, data={}):
        self.id = id
        self.weight = weight  # edge weight
        self.discount = discount  # maximum discount value for the robust case
        self.fail = fail  # whether edge failed
        self.success = True
        self.src = src  # source vertex
        self.src_id = src.id
        self.tgt = tgt  # target vertex
        self.tgt_id = tgt.id
        self.data = data

        # - edge-query-specific properties -
        self.p_reject = 0.0  # probability of rejection
        self.p_success_accept = 1.0  # probability of success, if accepted
        self.p_success_noquery = 1.0  # probability of success, if not queried

        self.sensitized = tgt.sensitized
        self.hash_str, self.hash_int = stable_hash((self.src_id, self.tgt_id, "Edge"))

    def __str__(self):
        return "V" + str(self.src.id) + "-V" + str(self.tgt.id)

    def __eq__(self, other):
        return self.hash_int == other.hash_int

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.hash_int

    def to_dict(self):
        e_dict = {
            "type": "pair_edge",
            "id": self.id,
            "weight": self.weight,
            "discount": self.discount,
            "src_id": self.src_id,
            "tgt_id": self.tgt_id,
            "sensitized": self.sensitized,
        }
        return e_dict

    def display(self, gamma):
        # if gamma == 0:
        #     return "src=%d, tgt=%d, weight=%f" % (self.src.id, self.tgt.id, self.weight)
        # else:
        return (
            "src=%d, tgt=%d, weight=%f, sens=%s, max_discount=%f, discount_frac=%f"
            % (
                self.src.id,
                self.tgt.id,
                self.weight,
                self.sensitized,
                self.discount,
                self.discount_frac,
            )
        )


class Digraph:
    """A directed graph, in which each edge has a numeric weight.

    Data members:
        n: the number of vertices in the digraph
        vs: an array of Vertex objects, such that vs[i].id == i
        es: an array of Edge objects, such that es[i].id = i
    """

    def __init__(self, n, aux_vertex_id=None):
        """Create a Digraph with n vertices"""
        self.n = n

        if aux_vertex_id is None:
            self.vs = [Vertex(i) for i in range(n)]
        else:
            self.vs = [Vertex(i, aux_id=aux_vertex_id[i]) for i in range(n)]

        self.adj_mat = [[None for x in range(n)] for x in range(n)]

        self.es = []
        self.cycles = []

    def add_edge(self, weight, source, tgt, edge_data={}):
        """Add an edge to the digraph

        Args:
            weight: the edge's weight, as a float
            source: the source Vertex
            tgt: the edge's target Vertex
        """

        id = len(self.es)
        e = Edge(id, weight, source, tgt, data=edge_data)
        self.es.append(e)
        source.edges.append(e)
        source.out_degree += 1
        tgt.in_degree += 1
        self.adj_mat[source.id][tgt.id] = e

    def remove_edge(self, e):
        self.es.remove(e)
        e.src.edges.remove(e)
        self.adj_mat[e.src.id][e.tgt.id] = None

    def find_cycles(self, max_length):
        """Find cycles of length up to max_length in the digraph.

        Returns:
            a list of cycles. Each cycle is represented as a list of
            vertices, with the first vertex _not_ repeated at the end.
        """
        cycle_list = [cycle for cycle in self.generate_cycles(max_length)]

        return cycle_list

    def write_cycles_to_file(self, cycle_file, max_length, cycle_list):
        cycle_ind_list = [[v.id for v in c] for c in cycle_list]
        # cycle_data = {'max_length': max_length, 'cycle_ind_list': cycle_ind_list}
        maxlen_data = {"max_length": max_length}
        # cycle_data = {'cycle_ind_list': cycle_ind_list}
        with open(cycle_file, "wb") as f:
            # json.dump(cycle_data, f, indent=4)
            f.write(json.dumps(max_length))
            f.write("\n")
            for c in cycle_list:  # in cycle_ind_list:
                f.write(json.dumps([v.id for v in c]))
                f.write("\n")

    def read_cycle_maxlen_from_file(self, cycle_file):
        with open(cycle_file, "rb") as f:
            max_length = json.loads(f.readline())
        return max_length

    def read_cycles_from_file(self, cycle_file, max_len):
        """
        Read cycle index lists from file, line by line.
        Only read cycles that are <= max_len
        """
        cycle_ind_list = []
        with open(cycle_file, "rb") as f:
            max_length = json.loads(f.readline())
            line = f.readline()
            while line:
                cycle_list = json.loads(line)
                if len(cycle_list) <= max_len:
                    cycle_ind_list.append(cycle_list)
                line = f.readline()

        return max_length, cycle_ind_list

    def generate_cycles(self, max_length):
        """Generate cycles of length up to max_length in the digraph.

        Each cycle yielded by this generator is represented as a list of
        vertices, with the first vertex _not_ repeated at the end.
        """

        vtx_used = [False] * len(
            self.vs
        )  # vtx_used[i]==True iff vertex i is in current path

        def cycle(current_path):
            last_vtx = current_path[-1]
            if self.edge_exists(last_vtx, current_path[0]):
                yield current_path[:]
            if len(current_path) < max_length:
                for e in last_vtx.edges:
                    v = e.tgt
                    if (
                        len(current_path) + shortest_paths_to_low_vtx[v.id]
                        <= max_length
                        and not vtx_used[v.id]
                    ):
                        current_path.append(v)
                        vtx_used[v.id] = True
                        for c in cycle(current_path):
                            yield c
                        vtx_used[v.id] = False
                        del current_path[-1]

        # Adjacency lists for transpose graph
        transp_adj_lists = [[] for v in self.vs]
        for edge in self.es:
            transp_adj_lists[edge.tgt.id].append(edge.src)

        for v in self.vs:
            shortest_paths_to_low_vtx = self.calculate_shortest_path_lengths(
                v,
                max_length - 1,
                lambda u: (w for w in transp_adj_lists[u.id] if w.id > v.id),
            )
            vtx_used[v.id] = True
            for c in cycle([v]):
                yield c
            vtx_used[v.id] = False

    def get_shortest_path_from_low_vtx(self, low_vtx, max_path):
        """ Returns an array of path lengths. For each v > low_vtx, if the shortest
            path from low_vtx to v is shorter than max_path, then element v of the array
            will be the length of this shortest path. Otherwise, element v will be
            999999999."""
        return self.calculate_shortest_path_lengths(
            self.vs[low_vtx],
            max_path,
            adj_list_accessor=lambda v: (e.tgt for e in v.edges if e.tgt.id >= low_vtx),
        )

    def get_shortest_path_to_low_vtx(self, low_vtx, max_path):
        """ Returns an array of path lengths. For each v > low_vtx, if the shortest
            path to low_vtx from v is shorter than max_path, then element v of the array
            will be the length of this shortest path. Otherwise, element v will be
            999999999."""

        def adj_list_accessor(v):
            for i in range(low_vtx, len(self.vs)):
                if self.adj_mat[i][v.id]:
                    yield self.vs[i]

        return self.calculate_shortest_path_lengths(
            self.vs[low_vtx], max_path, adj_list_accessor=adj_list_accessor
        )

    def calculate_shortest_path_lengths(
        self, from_v, max_dist, adj_list_accessor=lambda v: (e.tgt for e in v.edges)
    ):
        """Calculate the length of the shortest path from vertex from_v to each
        vertex with a greater or equal index, using paths containing
        only vertices indexed greater than or equal to from_v.

        Return value: a list of distances of length equal to the number of vertices.
        If the shortest path to a vertex is greater than max_dist, the list element
        will be 999999999.

        Args:
            from_v: The starting vertex
            max_dist: The maximum distance we're interested in
            adj_list_accessor: A function taking a vertex and returning an
                iterable of out-edge targets
        """
        # Breadth-first search
        q = deque([from_v])
        distances = [999999999] * len(self.vs)
        distances[from_v.id] = 0

        while q:
            v = q.popleft()
            # Note: >= is used instead of == on next line in case max_dist<0
            if distances[v.id] >= max_dist:
                break
            for w in adj_list_accessor(v):
                if distances[w.id] == 999999999:
                    distances[w.id] = distances[v.id] + 1
                    q.append(w)

        return distances

    def edge_exists(self, v1, v2):
        """Returns true if and only if an edge exists from Vertex v1 to Vertex v2."""

        return self.adj_mat[v1.id][v2.id] is not None

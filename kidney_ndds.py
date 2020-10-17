"""This module has objects for non-directed donors (NDDs) and chains initiated by NDDs.

In contrast to many kidney-exchange formulations, we do not include in the main directed
graph a vertex for each NDD. Rather, the program maintains a separate array
of Ndd objects. Each of these Ndd objects maintains a list of outgoing edges, and
each of these edges points towards a vertex (which represents a donor-patient pair)
in the directed graph.
"""

from kidney_digraph import KidneyReadException
from utils import stable_hash


class Ndd:
    """A non-directed donor"""

    def __init__(self, id=None, aux_id=None):
        self.edges = []
        self.discount_frac = 0
        self.chain_weight = 0
        self.id = id
        self.aux_id = aux_id
        self.out_degree = 0

    def add_edge(self, ndd_edge):
        """Add an edge representing compatibility with a patient who appears as a
        vertex in the directed graph."""
        self.out_degree += 1
        self.edges.append(ndd_edge)

    def remove_edge(self, ndd_edge):
        """Remove the edge."""
        self.edges.remove(ndd_edge)

    def get_edge(self, tgt_idx):
        """
        return the edge that points to vertex with id tgt_idx, if it exists
        """
        tgt_edges = [e for e in self.edges if e.tgt.id == tgt_idx]
        if len(tgt_edges) == 1:
            return tgt_edges[0]
        elif len(tgt_edges) == 0:
            raise Warning("edge not found to target index %d" % tgt_idx)
        else:
            raise Warning("multiple edges found to target index %d" % tgt_idx)

    def uniform_copy(self):
        """
        Return a copy of the ndd with all weights set to 1
        """
        n = Ndd(id=self.id)
        for e in self.edges:
            tgt = e.tgt
            new_weight = 1.0
            n.add_edge(NddEdge(tgt, new_weight, src_id=n.id, src=n))
        return n


class NddEdge:
    """An edge pointing from an NDD to a vertex in the directed graph"""

    def __init__(
        self,
        tgt,
        weight,
        discount=0,
        fail=False,
        discount_frac=0,
        src_id=None,
        src=None,
        data={},
    ):
        self.tgt = tgt
        self.src = src
        tgt.in_degree += 1
        self.tgt_id = tgt.id
        self.src_id = src_id
        self.weight = weight  # edge weight
        self.discount = discount  # discount value for the robust case
        self.fail = fail
        self.success = True
        self.discount_frac = discount_frac
        self.sensitized = tgt.sensitized
        self.data = data

        # - edge-query-specific properties -
        self.p_reject = 0.0  # probability of rejection
        self.p_success_accept = 1.0  # probability of success, if accepted
        self.p_success_noquery = 1.0  # probability of success, if not queried

        self.hash_str, self.hash_int = stable_hash(
            (self.src_id, self.tgt_id, "NddEdge")
        )

    def __str__(self):
        return "NDD edge to V{}".format(self.tgt.id)

    def __eq__(self, other):
        return self.hash_int == other.hash_int

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.hash_int

    def to_dict(self):
        e_dict = {
            "type": "ndd_edge",
            "weight": self.weight,
            "discount": self.discount,
            "discount_frac": self.discount_frac,
            "src_id": self.src_id,
            "tgt_id": self.tgt_id,
            "sensitized": self.sensitized,
        }
        return e_dict

    @classmethod
    def from_dict(cls, e_dict, ndds):
        # find NddEdge among provided ndds...
        # THIS DOESN'T HAVE TO BE A NddEdge FUNCTION

        # tgt = digraph.vs[e_dict['tgt_id']]
        # e = cls(tgt, e_dict['weight'], discount=e_dict['discount'], discount_frac=e_dict['discount_frac'],src_id = e_dict['src_id'])
        # return e

        e_tgt_id = e_dict["tgt_id"]
        for e in ndds[e_dict["src_id"]].edges:
            if e.tgt_id == e_tgt_id:
                return e
        raise Warning("NddEdge not found")

    def display(self):
        # if gamma == 0:
        #     return "NDD Edge: tgt=%d, weight=%f" % ( self.tgt.id, self.weight)
        # else:
        return (
            "NDD Edge: tgt=%d, weight=%f, sens=%s, max_discount=%f, discount_frac=%f"
            % (
                self.tgt.id,
                self.weight,
                self.sensitized,
                self.discount,
                self.discount_frac,
            )
        )


def create_relabelled_ndds(ndds, old_to_new_vtx):
    """Creates a copy of a n array of NDDs, with target vertices changed.

    If a target vertex in an original NDD had ID i, then the new target vertex
    will be old_to_new_vtx[i].
    """
    new_ndds = [Ndd(id=ndd.id) for ndd in ndds]
    for i, ndd in enumerate(ndds):
        for edge in ndd.edges:
            new_ndds[i].add_edge(
                NddEdge(
                    old_to_new_vtx[edge.tgt.id], edge.weight, src_id=ndd.id, src=ndd
                )
            )

    return new_ndds


def read_ndds(lines, digraph):
    """Reads NDDs from an array of strings in the .ndd format."""

    ndds = []
    ndd_count, edge_count = [int(x) for x in lines[0].split()]
    ndds = [Ndd(id=i) for i in range(ndd_count)]

    # Keep track of which edges have been created already so that we can
    # detect duplicates
    edge_exists = [[False for v in digraph.vs] for ndd in ndds]

    for line in lines[1 : edge_count + 1]:
        tokens = [t for t in line.split()]
        src_id = int(tokens[0])
        tgt_id = int(tokens[1])
        weight = float(tokens[2])
        if src_id < 0 or src_id >= ndd_count:
            raise KidneyReadException("NDD index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= digraph.n:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if edge_exists[src_id][tgt_id]:
            raise KidneyReadException(
                "Duplicate edge from NDD {0} to vertex {1}.".format(src_id, tgt_id)
            )
        ndds[src_id].add_edge(
            NddEdge(
                digraph.vs[tgt_id], weight, src_id=ndds[src_id].id, src=ndds[src_id]
            )
        )
        edge_exists[src_id][tgt_id] = True

    if lines[edge_count + 1].split()[0] != "-1" or len(lines) < edge_count + 2:
        raise KidneyReadException("Incorrect edge count")

    return ndds


class Chain(object):
    """A chain initiated by an NDD.
    
    Data members:
        ndd_index: The index of the NDD
        vtx_indices: The indices of the vertices in the chain, in order
        weight: the chain's weight
        edges: ordered list of the chain's edges
    """

    def __init__(self, ndd_index, vtx_indices, weight):  # , chain_edges):
        self.ndd_index = ndd_index
        self.vtx_indices = vtx_indices
        self.weight = weight
        self.discount_frac = 0.0
        # self.edges = chain_edges

    def to_dict(self):
        ch_dict = {
            "ndd_index": self.ndd_index,
            "vtx_indices": self.vtx_indices,
            "discount_frac": self.discount_frac,
            "weight": self.weight,
        }
        return ch_dict

    @property
    def length(self):
        return len(self.vtx_indices)

    @classmethod
    def from_dict(cls, ch_dict):
        ch = cls(ch_dict["ndd_index"], ch_dict["vtx_indices"], ch_dict["weight"])
        ch.discount_frac = ch_dict["discount_frac"]
        return ch

    def __repr__(self):
        return (
            "Chain NDD{} ".format(self.ndd_index)
            + " ".join(str(v) for v in self.vtx_indices)
            + " with weight "
            + str(self.weight)
        )

    def display(self):
        vtx_str = " ".join(str(v) for v in self.vtx_indices)
        return "Ndd %d, vtx = %s; weight = %f, discount_frac = %f" % (
            self.ndd_index,
            vtx_str,
            self.weight,
            self.discount_frac,
        )

    def __cmp__(self, other):
        # Compare on NDD ID, then chain length, then on weight, then
        # lexicographically on vtx indices
        if self.ndd_index < other.ndd_index:
            return -1
        elif self.ndd_index > other.ndd_index:
            return 1
        elif len(self.vtx_indices) < len(other.vtx_indices):
            return -1
        elif len(self.vtx_indices) > len(other.vtx_indices):
            return 1
        elif self.weight < other.weight:
            return -1
        elif self.weight > other.weight:
            return 1
        else:
            for i, j in zip(self.vtx_indices, other.vtx_indices):
                if i < j:
                    return -1
                elif i > j:
                    return 1
        return 0

    def get_edge_objs(self, digraph, ndds):
        """
        to get edges from a chain:
        first get the ndd edge into the digraph using ndd.get_edge(chain.vtx_indices[0])
        then get the edges in the graph object using graph.adj_mat
        """
        ndd = ndds[self.ndd_index]
        tgt_id = self.vtx_indices[0]

        edges = [ndd.get_edge(tgt_id)]

        for i_src in range(len(self.vtx_indices) - 1):
            curr_vtx = self.vtx_indices[i_src]
            next_vtx = self.vtx_indices[i_src + 1]
            edges.append(digraph.adj_mat[curr_vtx][next_vtx])
        return edges

    def get_weight(self, digraph, ndds, edge_success_prob):
        # chain weight is equal to e1.weight * p + e2.weight * p**2 + ... + en.weight * p**n
        ndd = ndds[self.ndd_index]
        # find the vertex that the NDD first donates to
        tgt_id = self.vtx_indices[0]
        e1 = []
        for e in ndd.edges:
            if e.tgt.id == tgt_id:
                # get edge...
                e1 = e
                break
        if e1 == []:
            print("chain.update_weight: could not find vertex id")
        weight = e1.weight * edge_success_prob  # add weight from ndd to first pair
        for j in range(len(self.vtx_indices) - 1):
            weight += digraph.adj_mat[self.vtx_indices[j]][
                self.vtx_indices[j + 1]
            ].weight * edge_success_prob ** (j + 2)
        return weight

    def weight_after_failure(self, digraph, ndds):
        ndd = ndds[self.ndd_index]
        # find the vertex that the NDD first donates to
        tgt_id = self.vtx_indices[0]
        e1 = []
        for e in ndd.edges:
            if e.tgt.id == tgt_id:
                # get edge...
                e1 = e
                break
        if e1 == []:
            raise Warning("could not find vertex id")
        weight = 0
        if e1.fail:
            return weight
        else:
            weight += e1.weight  # add weight from ndd to first pair
            for j in range(len(self.vtx_indices) - 1):
                if digraph.adj_mat[self.vtx_indices[j]][self.vtx_indices[j + 1]].fail:
                    return weight
                else:
                    weight += digraph.adj_mat[self.vtx_indices[j]][
                        self.vtx_indices[j + 1]
                    ].weight
        return weight

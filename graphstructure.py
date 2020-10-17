import kidney_utils
from kidney_digraph import Cycle, Digraph
from kidney_ip import OptConfig, create_picef_model
from kidney_ndds import Ndd, NddEdge
from kpd_data import KPDData

import numpy as np

from utils import get_logger, initialize_random_edge_weights

logger = get_logger()


class GraphStructure(object):
    """contains an instance of the kidney exchange problem. also includes matching parameters (cycle/chain cap) and
    edge failure probabilities used when constructing the matching."""

    def __init__(
        self,
        graph,
        altruists,
        cycle_cap,
        chain_cap,
        name=None,
        df_donor=None,
        df_recip=None,
    ):

        self.graph = graph
        self.altruists = altruists
        self.cycle_cap = cycle_cap
        self.chain_cap = chain_cap
        self.all_edge_list = list(self.all_edges())
        self.name = name
        self.df_donor = df_donor
        self.df_recip = df_recip
        self.init_edge_ids()
        self.init_vertex_ids()
        self.init_matchable()
        self.matchable_edge_list = [e for e in self.all_edge_list if e.matchable]

        self.init_optconfig()

    @classmethod
    def er_randomgraph(
        cls, num_vertices, p, seed, cycle_cap=3, chain_cap=4
    ):
        """
        generate an Erdos-Renyi random graph with a specified number of vertices and edge probability. all edge weights
        are 1

        if any vertices are disconnected, they are removed.
        """

        rs = np.random.RandomState(seed)

        name = f"random_n_{num_vertices}_p_{p}_s_{seed}"

        weight = 1.0

        # generate random adjmat
        # adjmat[i, j] is an edge from vertex i to vertex j
        adjmat = rs.choice([1, 0], (num_vertices, num_vertices), p=[p, 1 - p])

        # set the diagonal to 0
        for i in range(num_vertices):
            adjmat[i, i] = 0

        in_deg = [adjmat[:, i].sum() for i in range(num_vertices)]
        out_deg = [adjmat[i, :].sum() for i in range(num_vertices)]

        # any vertices with zero in degree is an ndd, all others are pair vertices
        ndd_inds = [
            i for i in range(num_vertices) if ((in_deg[i] == 0) and out_deg[i] > 0)
        ]
        all_vertex_ids = set(range(num_vertices)).difference(ndd_inds)
        vertex_inds = list(
            filter(lambda i: in_deg[i] > 0, all_vertex_ids)
        )  # take only vertices with incoming edges

        # take an id (0 to num_vertices + 1) to a index (0 to len(vertex_inds))
        vtx_id_to_ind = {j: i for i, j in enumerate(vertex_inds)}

        # create ndds
        ndd_list = [Ndd(id=i) for i in range(len(ndd_inds))]

        # create digraph
        digraph = Digraph(len(vertex_inds))

        # add ndd edges
        for i_ndd, ndd_ind in enumerate(ndd_inds):
            for recip_id in adjmat[i_ndd, :].nonzero()[0]:
                ndd_list[i_ndd].add_edge(
                    NddEdge(
                        digraph.vs[vtx_id_to_ind[recip_id]],
                        weight,
                        src_id=ndd_list[i_ndd].id,
                        src=ndd_list[i_ndd],
                    )
                )

        # add pair-pair edges
        for v_ind in vertex_inds:
            src_ind = vtx_id_to_ind[v_ind]
            for recip_id in adjmat[src_ind, :].nonzero()[0]:
                digraph.add_edge(
                    weight, digraph.vs[src_ind], digraph.vs[vtx_id_to_ind[recip_id]]
                )
        graph = cls(
            digraph, ndd_list, cycle_cap=cycle_cap, chain_cap=chain_cap, name=name
        )

        # set random edge weights
        initialize_random_edge_weights(graph, rs)

        return graph

    def init_optconfig(self, edge_success_prob=1.0):
        # initialize OptConfig for this graph
        self.optconfig = OptConfig(
            self.graph,
            self.altruists,
            self.cycle_cap,
            self.cycle_cap,
            edge_success_prob=edge_success_prob,
        )
        # add the gurobi model to the OptConfig object
        create_picef_model(self.optconfig)

    def init_matchable(self):
        """initialize the property matchable for each edge in the graph. matchable = True means the edge can be included
        in any matching (i.e., it's in at least one cycle or chain)"""

        # initialize all edges to be not-matchable
        for e in self.all_edge_list:
            e.matchable = False

        cycles = self.graph.find_cycles(self.cycle_cap)
        cycle_list = []
        # all edges in a cycle are matchable
        for c in cycles:
            c_obj = Cycle(c)
            c_obj.add_edges(self.graph.es)
            cycle_list.append(c_obj)
            for e in c_obj.edges:
                e.matchable = True

        # all NDDEdges are matchable
        for n in self.altruists:
            for e in n.edges:
                e.matchable = True

        # all pair-pair edges that are within chain distance are matchable
        dists_from_ndd = kidney_utils.get_dist_from_nearest_ndd(
            self.graph, self.altruists
        )
        for e in self.all_edge_list:
            if dists_from_ndd[e.src.id] <= self.chain_cap - 1:
                e.matchable = True

        logger.info(
            f"matchable edges: {len([e for e in self.all_edge_list if e.matchable])} (out of {len(self.all_edge_list)} total)"
        )

    def get_kpd_data(self):
        assert self.df_donor is not None
        assert self.df_recip is not None

        kpd_data = KPDData()

        for v in self.graph.vs:
            # count all donor and recipient abo types. this adds N pairs for each recip associated with N donors
            recip_abo = self.lookup_recip_abo(v.aux_id)
            recip_sens = self.lookup_recip_sensitized(v.aux_id)

            if recip_sens == "Y":
                kpd_data.high_low_sensitized_count[0] += 1
            if recip_sens == "N":
                kpd_data.high_low_sensitized_count[1] += 1

            kpd_data.recip_abo[recip_abo] += 1
            for donor_id in v.donor_set:
                donor_abo = self.lookup_donor_abo(donor_id)
                kpd_data.donor_abo[donor_abo] += 1
                kpd_data.pair_abo[(recip_abo, donor_abo)] += 1

        # get bin counts for in/out degree
        in_deg_list = [v.in_degree for v in self.graph.vs]
        out_deg_list = [v.out_degree for v in self.graph.vs]
        ndd_out_deg_list = [ndd.out_degree for ndd in self.altruists]
        kpd_data.in_deg_counts = np.histogram(
            in_deg_list, bins=KPDData.in_deg_bin_edges
        )[0]
        kpd_data.out_deg_counts = np.histogram(
            out_deg_list, bins=KPDData.out_deg_bin_edges
        )[0]
        kpd_data.ndd_out_deg_counts = np.histogram(
            ndd_out_deg_list, bins=KPDData.out_deg_bin_edges
        )[0]

        for ndd in self.altruists:
            ndd_donor_abo = self.lookup_donor_abo(ndd.aux_id)  # should work?
            kpd_data.ndd_abo[ndd_donor_abo] += 1

        return kpd_data

    def lookup_recip_abo(self, patient_id):
        """look up recipient abo in graph.df_recip, using the recipient's unos id"""
        assert self.df_recip is not None
        return self.df_recip[self.df_recip["kpd_candidate_id"] == patient_id][
            "abo"
        ].values[0]

    def lookup_recip_sensitized(self, patient_id):
        """look up recipient highly sensitized status in graph.df_recip, using the recipient's unos id"""
        assert self.df_recip is not None
        return self.df_recip[self.df_recip["kpd_candidate_id"] == patient_id][
            "highly_sensitized"
        ].values[0]

    def lookup_donor_abo(self, donor_id):
        """look up recipient abo in graph.df_donor, using the donor's unos id"""
        assert self.df_donor is not None
        return self.df_donor[self.df_donor["kpd_donor_id"] == donor_id]["abo"].values[0]

    def init_edge_ids(self):
        """initialize edge index property to be unique for each edge in the graph"""
        for i, edge in enumerate(self.all_edge_list):
            edge.index = i

    def init_vertex_ids(self):
        """initialize vertex index property to be unique for each edge in the graph"""
        for i, ndd in enumerate(self.altruists):
            ndd.index = i
        for i, v in enumerate(self.graph.vs):
            v.index = i + len(self.altruists)

    def all_edges(self):
        """yield edges in order, first digraph then ndd"""
        for edge in self.graph.es:
            yield edge
        for ndd in self.altruists:
            for edge in ndd.edges:
                yield edge

    def all_edges_after(self, index):
        """returns all edges after the specified index"""
        return self.all_edge_list[index + 1 :]

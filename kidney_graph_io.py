import glob
import logging
import os
import pandas as pd

from graphstructure import GraphStructure
from kidney_digraph import Digraph, KidneyReadException
from kidney_ndds import Ndd, NddEdge
from utils import simple_string

FORMAT = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger()

NULL_KPD_DATA = None


def read_unos_graph(directory, cycle_cap, chain_cap):
    """read a unos-format exchange, and return a list of kidney_ndd.Ndd objects and a kidney_digraph.Digraph object.

    each unos-format exchange is contained in a subdirectory with the naming format 'KPD_CSV_IO_######'. Each exchange
     subdirectory must contain a file of the format ########_edgeweights.csv
    """
    # look for edge files
    edge_files = glob.glob(os.path.join(directory, "*edgeweights.csv"))

    name = os.path.basename(directory)

    # there should only be one edgeweights file
    assert len(edge_files) == 1

    edge_filename = edge_files[0]

    df = pd.read_csv(edge_filename)

    assert df.columns == [
        "KPD Match Run ID",
        "KPD Candidate ID",
        "Candidate's KPD Pair ID",
        "KPD Donor ID",
        "Donor's KPD Pair ID",
        "Total Weight",
    ]

    col_names = [
        "match_run",
        "patient_id",
        "patient_pair_id",
        "donor_id",
        "donor_paired_patient_id",
        "weight",
    ]

    df.columns = col_names

    # last column is edge weights -- only take nonzero edges
    nonzero_edges = df.loc[df["weight"] > 0]

    # remove NDD edges
    kpd_edges = nonzero_edges.loc[~nonzero_edges["donor_paired_patient_id"].isnull()]

    # get unique vertex ids
    # Note, in the *edgeweights.csv files:
    # - "KPD Candidate ID" (or "patient_id" here) is the patient/recipient's UNOS ID
    # - "Donor's KPD Pair ID" is the UNOS ID of the donor's associated patient (or None if the donor is an NDD)
    vtx_id = set(
        list(kpd_edges["patient_id"].unique())
        + list(kpd_edges["donor_paired_patient_id"].unique())
    )

    vtx_count = len(vtx_id)
    digraph = Digraph(vtx_count)

    # vtx_index[id] gives the index in the digraph
    vtx_index = dict(zip(vtx_id, range(len(vtx_id))))

    warned = False
    for index, row in kpd_edges.iterrows():
        src_id = vtx_index[row["donor_paired_patient_id"]]
        tgt_id = vtx_index[row["patient_id"]]
        weight = row["weight"]
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if src_id == tgt_id:
            raise KidneyReadException(
                "Self-loop from {0} to {0} not permitted".format(src_id)
            )
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]) & ~warned:
            print("# WARNING: Duplicate edge in file: {}".format(edge_filename))
            warned = True
        if weight == 0:
            raise KidneyReadException(
                "Zero-weight edge from {} to {}".format(src_id, tgt_id)
            )

        digraph.add_edge(weight, digraph.vs[src_id], digraph.vs[tgt_id])

    # now read NDDs - take only NDD edges
    ndd_edges = nonzero_edges.loc[nonzero_edges["donor_paired_patient_id"].isnull()]
    ndd_id = set(list(ndd_edges["donor_id"].unique()))

    ndd_count = len(ndd_id)

    if ndd_count > 0:
        ndd_list = [Ndd(id=i) for i in range(ndd_count)]
        ndd_index = dict(
            zip(ndd_id, range(len(ndd_id)))
        )  # ndd_index[id] gives the index in the digraph

        # Keep track of which edges have been created already, to detect duplicates
        edge_exists = [[False for v in digraph.vs] for ndd in ndd_list]

        for index, row in ndd_edges.iterrows():
            src_id = ndd_index[row["donor_id"]]
            tgt_id = vtx_index[row["patient_pair_id"]]
            weight = row["weight"]
            if src_id < 0 or src_id >= ndd_count:
                raise KidneyReadException("NDD index {} out of range.".format(src_id))
            if tgt_id < 0 or tgt_id >= digraph.n:
                raise KidneyReadException(
                    "Vertex index {} out of range.".format(tgt_id)
                )

            ndd_list[src_id].add_edge(
                NddEdge(
                    digraph.vs[tgt_id],
                    weight,
                    src_id=ndd_list[src_id].id,
                    src=ndd_list[src_id],
                )
            )
            edge_exists[src_id][tgt_id] = True
    else:
        ndd_list = []
        ndd_index = []

    graph = GraphStructure(digraph, ndd_list, cycle_cap, chain_cap, name=name)

    return graph


def read_unos_graph_with_data(directory, cycle_cap, chain_cap):
    """
    read a unos-format exchange, and return a list of kidney_ndd.Ndd objects and a kidney_digraph.Digraph object, and
    save donor/recipient data

    each unos-format exchange is contained in a subdirectory with the naming format 'KPD_CSV_IO_######'. Each exchange
     subdirectory must contain files of the format:
      - *edgeweights.csv
      - *donor.csv
      - *recipient.csv
    """

    if directory.endswith(os.sep):
        name = os.path.basename(directory[:-1])
    else:
        name = os.path.basename(directory)

    # look for  files
    edge_files = glob.glob(os.path.join(directory, "*edgeweights.csv"))
    donor_files = glob.glob(os.path.join(directory, "*donor.csv"))
    recip_files = glob.glob(os.path.join(directory, "*recipient.csv"))

    # there should only be one of each file
    assert len(donor_files) == 1
    assert len(recip_files) == 1
    assert len(edge_files) == 1

    donor_file = donor_files[0]
    recip_file = recip_files[0]
    edge_filename = edge_files[0]

    df_donor = pd.read_csv(donor_file)
    df_recip = pd.read_csv(recip_file)

    # make all cols lowercase
    df_donor.columns = [c.lower() for c in df_donor.columns]
    df_recip.columns = [c.lower() for c in df_recip.columns]

    # if no cpra col, then add null values
    if "cpra" not in df_recip.columns:
        logger.info("CPRA column not found")
        df_recip["cpra"] = NULL_KPD_DATA

    # -- add columns for missing data if they don't exist
    # if no cpra col, then add null values
    if "cpra" not in df_recip.columns:
        logger.info("COL NOT FOUND: cpra")
        df_recip["cpra"] = NULL_KPD_DATA

    if "highly_sensitized" not in df_recip.columns:
        logger.info("COL NOT FOUND: highly_sensitized")
        df_recip["highly_sensitized"] = NULL_KPD_DATA

    if "abo" not in df_recip.columns:
        if "abo blood group" in df_recip.columns:
            df_recip["abo"] = df_recip["abo blood group"]
        else:
            raise Exception("no abo column found")

    # validate donor data
    assert "abo" in df_donor.columns

    # validate recip data
    assert "abo" in df_recip.columns
    assert "cpra" in df_recip.columns
    assert "highly_sensitized" in df_recip.columns

    # remove abo subtypes and make lowercase
    df_donor["abo"] = df_donor["abo"].apply(
        lambda x: simple_string(x, non_numeric=True)
    )
    df_recip["abo"] = df_recip["abo"].apply(
        lambda x: simple_string(x, non_numeric=True)
    )

    df_edges = pd.read_csv(edge_filename)

    expected_columns = [
        "KPD Match Run ID",
        "KPD Candidate ID",
        "Candidate's KPD Pair ID",
        "KPD Donor ID",
        "Donor's KPD Pair ID",
        "Total Weight",
    ]

    assert len(expected_columns) == len(df_edges.columns)

    for i_col, expected in enumerate(expected_columns):
        assert simple_string(expected) == simple_string(df_edges.columns[i_col])

    col_names = [
        "match_run",
        "patient_id",
        "patient_pair_id",
        "donor_id",
        "donor_paired_patient_id",
        "weight",
    ]

    df_edges.columns = col_names

    # last column is edge weights -- only take nonzero edges
    nonzero_edges = df_edges.loc[df_edges["weight"] > 0]

    # remove NDD edges
    kpd_edges = nonzero_edges.loc[~nonzero_edges["donor_paired_patient_id"].isnull()]

    # get unique vertex ids
    # Note, in the *edgeweights.csv files:
    # - "KPD Candidate ID" (or "patient_id" here) is the patient/recipient's UNOS ID
    # - "Donor's KPD Pair ID" is the UNOS ID of the donor's associated patient (or None if the donor is an NDD)
    vtx_id = set(
        list(kpd_edges["patient_id"].unique())
        + list(kpd_edges["donor_paired_patient_id"].unique())
    )

    # vtx_index[id] gives the index in the digraph
    vtx_count = len(vtx_id)
    vtx_index = dict(zip(vtx_id, range(len(vtx_id))))
    vtx_index_to_id = {v: k for k, v in vtx_index.items()}

    digraph = Digraph(vtx_count, aux_vertex_id=vtx_index_to_id)

    warned = False
    for index, row in kpd_edges.iterrows():
        src_id = vtx_index[row["donor_paired_patient_id"]]
        tgt_id = vtx_index[row["patient_id"]]
        weight = row["weight"]
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if src_id == tgt_id:
            raise KidneyReadException(
                "Self-loop from {0} to {0} not permitted".format(src_id)
            )
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]) & ~warned:
            print("# WARNING: Duplicate edge in file: {}".format(edge_filename))
            warned = True
        if weight == 0:
            raise KidneyReadException(
                "Zero-weight edge from {} to {}".format(src_id, tgt_id)
            )

        # for the donor pair, add the the donor ID to the vertex's list of donor IDs unless it's already there
        digraph.vs[src_id].donor_set.add(row["donor_id"])

        digraph.add_edge(
            weight, digraph.vs[src_id], digraph.vs[tgt_id], edge_data=row.to_dict()
        )

    # now read NDDs - take only NDD edges
    ndd_edges = nonzero_edges.loc[nonzero_edges["donor_paired_patient_id"].isnull()]
    ndd_id = set(list(ndd_edges["donor_id"].unique()))

    ndd_count = len(ndd_id)

    if ndd_count > 0:
        ndd_list = [Ndd(id=i) for i in range(ndd_count)]
        ndd_index = dict(
            zip(ndd_id, range(len(ndd_id)))
        )  # ndd_index[id] gives the index in the digraph

        for k, v in ndd_index.items():
            ndd_list[v].aux_id = k

        # Keep track of which edges have been created already, to detect duplicates
        edge_exists = [[False for v in digraph.vs] for ndd in ndd_list]

        for index, row in ndd_edges.iterrows():
            src_id = ndd_index[row["donor_id"]]
            tgt_id = vtx_index[row["patient_pair_id"]]
            weight = row["weight"]
            if src_id < 0 or src_id >= ndd_count:
                raise KidneyReadException("NDD index {} out of range.".format(src_id))
            if tgt_id < 0 or tgt_id >= digraph.n:
                raise KidneyReadException(
                    "Vertex index {} out of range.".format(tgt_id)
                )

            ndd_list[src_id].add_edge(
                NddEdge(
                    digraph.vs[tgt_id],
                    weight,
                    src_id=ndd_list[src_id].id,
                    src=ndd_list[src_id],
                )
            )
            edge_exists[src_id][tgt_id] = True
    else:
        ndd_list = []

    graph = GraphStructure(
        digraph,
        ndd_list,
        cycle_cap,
        chain_cap,
        name=name,
        df_donor=df_donor,
        df_recip=df_recip,
    )

    return graph


def read_cmu_format(details_filename, maxcard_filename, drop_edge_frac=0.0, seed=101):
    # read a "cmu" format exchange graph, using the details and maxcard files
    #
    # optional : drop_edge_frac in (0, 1) removes a fraction of the edges to the Digraph.

    name = os.path.basename(maxcard_filename)

    # read details.input file
    col_names = [
        "id",
        "abo_patient",
        "abo_fonor",
        "wife_patient",
        "pra",
        "in_deg",
        "out_deg",
        "is_ndd",
        "is_marginalized",
    ]
    df_details = pd.read_csv(
        details_filename, names=col_names, skiprows=1, delim_whitespace=True
    )

    pair_details = df_details.loc[df_details["is_ndd"] == 0]
    pair_id = list(pair_details["id"].unique())

    # vtx_index[id] gives the index in the digraph
    vtx_index = dict(zip(pair_id, range(len(pair_id))))

    vtx_count = len(vtx_index)
    digraph = Digraph(vtx_count)

    # label sensitized pairs
    for index, row in pair_details.iterrows():
        if row["is_marginalized"]:
            digraph.vs[vtx_index[row["id"]]].sensitized = True

    # read maxcard.inuput file (edges)
    col_names = ["src_id", "tgt_id", "weight", "c4", "c5"]
    df_edges = pd.read_csv(
        maxcard_filename, names=col_names, skiprows=1, delim_whitespace=True
    )

    # drop the last column
    df_edges.drop(df_edges.index[-1])

    # take only nonzero edges
    nonzero_edges = df_edges.loc[df_edges["weight"] > 0]

    # optional: sample from the edges
    if drop_edge_frac != 0.0:
        assert (drop_edge_frac < 1.0) and (drop_edge_frac > 0.0)
        nonzero_edges = nonzero_edges.sample(
            frac=(1.0 - drop_edge_frac), random_state=seed
        )

    # ind ndds if they exist
    ndd_details = df_details.loc[df_details["is_ndd"] == 1]
    ndd_count = len(ndd_details)

    if ndd_count > 0:
        ndd_list = [Ndd(id=i) for i in range(ndd_count)]
        ndd_id = list(ndd_details["id"].unique())

        # ndd_index[id] gives the index in the ndd list
        ndd_index = dict(zip(ndd_id, range(len(ndd_id))))
    else:
        ndd_list = []
        ndd_index = []

    use_ndds = ndd_count > 0

    # add edges to pairs and ndds
    for index, row in nonzero_edges.iterrows():
        src = row["src_id"]
        tgt_id = vtx_index[row["tgt_id"]]
        weight = row["weight"]
        if use_ndds and (src in ndd_index.keys()):  # this is an ndd edge
            src_id = ndd_index[src]
            ndd_list[src_id].add_edge(
                NddEdge(
                    digraph.vs[tgt_id],
                    weight,
                    src_id=ndd_list[src_id].id,
                    src=ndd_list[src_id],
                )
            )
        else:  # this edge is a pair edge
            src_id = vtx_index[src]
            digraph.add_edge(weight, digraph.vs[src_id], digraph.vs[tgt_id])

    return digraph, ndd_list, name


def get_cmu_graphs(
    directory, cycle_cap, chain_cap, drop_edge_frac=0.0, seed=0,
):
    # create a generator that produces kidney exchange graphs, given a directory containing "cmu" format exchange
    # graph files.

    # find all *maxcard.input files in the directory -- each corresponds to an exchange graph
    maxcard_files = glob.glob(os.path.join(directory, "*maxcard.input"))

    for maxcard_file in maxcard_files:
        graph = get_single_cmu_graph(
            maxcard_file,
            cycle_cap,
            chain_cap,
            drop_edge_frac=drop_edge_frac,
            seed=seed,
        )
        yield graph


def get_single_cmu_graph(
    maxcard_file, cycle_cap, chain_cap, drop_edge_frac=0.0, seed=0,
):

    file_base = "_".join(maxcard_file.split("_")[:-1])

    # find the details file; there can be only one
    details_files = glob.glob(file_base + "_*details.input")
    assert len(details_files) == 1
    details_file = details_files[0]

    digraph, ndd_list, name = read_cmu_format(
        details_file, maxcard_file, drop_edge_frac=drop_edge_frac, seed=seed
    )

    graph = GraphStructure(digraph, ndd_list, cycle_cap, chain_cap, name=name)

    return graph


def get_unos_graphs(directory, cycle_cap, chain_cap):
    # create a generator that produces kidney exchange graphs, given a directory containing "unos" format exchange
    # graph files.

    # find all directories in this directories
    unos_dirs = glob.glob(os.path.join(directory, "*/"))

    for unos_dir in unos_dirs:
        graph = read_unos_graph_with_data(unos_dir, cycle_cap, chain_cap)
        yield graph

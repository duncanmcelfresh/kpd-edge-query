# kpd-edge-query

Codebase accompanying the paper *Improving Policy-Constrained Kidney Exchange via Pre-Screening* (NeurIPS 20), by Duncan C McElfresh, Michael Curry, Tuomas Sandholm, and John P Dickerson.

### overview

Files ``full_experiment.py`` and ``greedy_exhaustive_check.py`` run the main experiments for our paper. 

Files ``kidney_digraph.py``, ``kidney_ip.py``, ``kidney_ndd.py``, ``kidney_utils.py``, and ``graphstructure.py`` contain all code related to the kidney exchange clearing problem. The first four of these files are adapted (heavily) from James Trimble's [kidney_solver](https://github.com/jamestrimble/kidney_solver) repo.

Files ``edge_selection_tree.py``, ``greedy_edge_choice.py``, and ``multistage_edge_selection.py`` implement the edge selection methods described in our paper, and evaluate the objective of the edge query problem.
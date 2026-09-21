# Risk detection plan

Every critical risk in `risk_matrix.csv` is tied to an observable diagnostic and a kill condition. The order is: validate flow-disjoint data manifests; measure 1-5 step errors; measure counterfactual candidate ranking; calibrate support/uncertainty; then run controller comparisons. Control metrics without these diagnostics cannot establish a World Model mechanism. External-paper numbers are contextual only until rerun under one simulator contract.

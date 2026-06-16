"""Labor-law ZK (Zero-Knowledge Proof) module.

This module adapts the shared TruthFinder circom circuit (v2, top1-in-circuit)
to the labour-law domain.  It provides:

1. ``zk_labor_builder`` — labour-law-specific ZK input construction and
   circuit-reference verification.
2. ``TruthFinder_circuit_ref`` — Python implementation of the circuit semantics
   (copied from the translation-app ZK module; identical circuit).
3. ``prepare_circom_input`` / ``expander`` — generic dense → circom conversion
   utilities (also shared with the translation-app ZK module).
"""

from labor_law_app.zk.zk_labor_builder import (  # noqa: F401
    K_MAX_FIXED,
    M_FIXED,
    N_MAX_FIXED,
    Q16,
    ITER_N_FIXED,
    FIXED_PARAMS_Q16,
    build_labor_circom_input,
    collect_labor_zk_state,
    run_full_labor_zk_pipeline,
    run_labor_zk_verification,
)

"""Blind evaluation harness for the OASiS campaign.

Separation of roles, enforced structurally:

    builder  (offline)  sees u_exact -- derives and verifies the source term
    agent    (runtime)  NEVER sees it -- solves the stated problem
    grader   (offline)  opens the sealed key -- after the fact

Modules:
    spec      closed-field container a prompt is built from (structural blindness)
    derive    u_exact -> source term, with proof and blindness design checks
    leakgate  does a task text disclose the solution?  (symbolic, not lexical)
    selfconv  observed order from mesh halving alone -- no exact solution
    keyvault  encryption, filesystem sealing, SHA-256 commitment
"""

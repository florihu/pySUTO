# pySUTO

**Python Supply–Use Table Optimiser**

A modular Python framework for constructing, reconciling, and propagating uncertainty through material flow accounts derived from heterogeneous data sources. pySUTO takes raw statistical data feeds, maps them onto a common accounting structure, aggregates across sources with inverse-variance weighting, imputes missing values, builds a constrained linear system, and solves a σ-weighted QP — tracking uncertainty end-to-end from raw pedigree scores to the final optimum.

see [CONTEXT.md] for further information
"""DC-PSR: Degradation-Consistent Probabilistic Stage Representation.

Dataset-independent implementation.

Architecture
------------
    DatasetAdapter  ->  Shared DC-PSR Pipeline  ->  Experiment Runner

The adapter is the ONLY dataset-specific part. It must supply a run-level
feature table with a fixed set of meta columns (see datasets/base.py) plus an
arbitrary number of numeric signal-feature columns, and a list of task
definitions. Everything downstream is dataset-agnostic.
"""
__version__ = "1.0.0"

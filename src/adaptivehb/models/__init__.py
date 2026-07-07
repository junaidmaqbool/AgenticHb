"""Model implementations.

Currently provides dummy trainables for framework dry-runs; real segmentation
and prediction models are added in Phases 5-6 behind the same ``Trainable``
contract.
"""

from adaptivehb.models.dummy import DummyTrainable, make_dummy_factory

__all__ = ["DummyTrainable", "make_dummy_factory"]

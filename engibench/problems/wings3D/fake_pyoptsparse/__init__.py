"""Drop-in module for pyoptsparse to unpickle ahistory when pyoptsparse is not installed."""

from types import ModuleType


class FakePyOptSparseObject:
    """Drop-in for objects needed to unpickle a pyoptsparse history when pyoptsparse is not installed."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def _mapContoOpt_Dict(self, d):  # noqa: N802
        return d

    def _mapXtoOpt_Dict(self, d):  # noqa: N802
        return d

    def _mapObjtoOpt_Dict(self, d):  # noqa: N802
        return d


class Optimization(FakePyOptSparseObject):
    """Drop-in."""


class Variable(FakePyOptSparseObject):
    """Drop-in."""


class Constraint(FakePyOptSparseObject):
    """Drop-in."""


class Objective(FakePyOptSparseObject):
    """Drop-in."""

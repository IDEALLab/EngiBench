from dataclasses import dataclass

from engibench.utils.upcast import upcast


@dataclass
class X:
    a: int = 1
    b: float = 2.0


class Y(X):
    c: bool = False


class Z(Y):
    d: str = "..."


def test_upcast_works_for_direct_super_class() -> None:
    y = Y()
    x = upcast(y)

    assert x == X()


def test_upcast_works_for_indirect_super_class() -> None:
    z = Z()
    x = upcast(z, X)
    x2 = upcast(upcast(z))

    assert x == X()
    assert x2 == X()

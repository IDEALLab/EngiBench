"""Cast dataclass instances to its super type."""

import dataclasses
from typing import Any


def upcast(obj: Any, super_type: Any | None = None) -> Any:
    """Restrict the fields of a dataclass instance to the fields of its super type."""
    if super_type is None:
        super_type = type(obj).mro()[1]
    assert dataclasses.is_dataclass(obj)
    assert not isinstance(obj, type)
    assert dataclasses.is_dataclass(super_type)
    super_fields: set[str] = {f.name for f in dataclasses.fields(super_type)}
    return super_type(**{key: val for key, val in dataclasses.asdict(obj).items() if key in super_fields})

from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass
class ColumnSchema:
    dtype: str  # "int", "float", "str", "date", "bool"
    nullable: bool = True
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""

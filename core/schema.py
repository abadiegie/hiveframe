from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Callable, Optional

import pandas as pd

@dataclass
class ColumnSchema:
    dtype: str  # "int", "float", "str", "date", "bool"
    nullable: bool = True
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    coerce: bool = False

    _SUPPORTED_DTYPES = {"int", "float", "str", "date", "bool"}

    def __post_init__(self) -> None:
        self.dtype = self.dtype.lower()
        if self.dtype not in self._SUPPORTED_DTYPES:
            supported = ", ".join(sorted(self._SUPPORTED_DTYPES))
            raise ValueError(f"Unsupported schema dtype '{self.dtype}'. Expected one of: {supported}")

    def validate(self, value: Any, column_name: str) -> Any:
        if value is None:
            if not self.nullable:
                raise ValueError(f"Column '{column_name}' is not nullable")
            return None

        normalized = self._coerce_value(value, column_name) if self.coerce else value

        if self.dtype == "int" and not self._is_int(normalized):
            raise TypeError(f"Column '{column_name}' expects int, got {type(normalized).__name__}")
        if self.dtype == "float" and not self._is_float(normalized):
            raise TypeError(f"Column '{column_name}' expects float, got {type(normalized).__name__}")
        if self.dtype == "str" and not isinstance(normalized, str):
            raise TypeError(f"Column '{column_name}' expects str, got {type(normalized).__name__}")
        if self.dtype == "bool" and not self._is_bool(normalized):
            raise TypeError(f"Column '{column_name}' expects bool, got {type(normalized).__name__}")
        if self.dtype == "date" and not self._is_date_like(normalized):
            raise TypeError(f"Column '{column_name}' expects date, got {type(normalized).__name__}")

        if self.validator and not self.validator(normalized):
            raise ValueError(f"Column '{column_name}' failed custom validation: {normalized}")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "nullable": self.nullable,
            "description": self.description,
            "coerce": self.coerce,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ColumnSchema":
        return cls(
            dtype=str(payload["dtype"]),
            nullable=bool(payload.get("nullable", True)),
            description=str(payload.get("description", "")),
            coerce=bool(payload.get("coerce", False)),
        )

    def _coerce_value(self, value: Any, column_name: str) -> Any:
        try:
            if self.dtype == "int":
                return self._coerce_int(value)
            if self.dtype == "float":
                return self._coerce_float(value)
            if self.dtype == "str":
                return str(value)
            if self.dtype == "bool":
                return self._coerce_bool(value)
            if self.dtype == "date":
                return self._coerce_date(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Column '{column_name}' expects {self.dtype}, got {type(value).__name__}") from exc
        return value

    @classmethod
    def _coerce_int(cls, value: Any) -> int:
        if cls._is_bool(value):
            raise TypeError("bool is not a valid int value")
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            return int(value.strip())
        raise TypeError("value is not int-like")

    @classmethod
    def _coerce_float(cls, value: Any) -> float:
        if cls._is_bool(value):
            raise TypeError("bool is not a valid float value")
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        raise TypeError("value is not float-like")

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            truthy = {"1", "true", "yes", "y", "on"}
            falsy = {"0", "false", "no", "n", "off"}
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
            raise TypeError("value is not bool-like")
        if isinstance(value, Integral) and not isinstance(value, bool):
            if value in (0, 1):
                return bool(value)
        if isinstance(value, bool):
            return value
        raise TypeError("value is not bool-like")

    @staticmethod
    def _coerce_date(value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return pd.Timestamp(value)
        raise TypeError("value is not date-like")

    @staticmethod
    def _is_bool(value: Any) -> bool:
        return isinstance(value, (bool, pd.BooleanDtype().type))

    @classmethod
    def _is_int(cls, value: Any) -> bool:
        if cls._is_bool(value):
            return False
        return isinstance(value, Integral)

    @classmethod
    def _is_float(cls, value: Any) -> bool:
        if cls._is_bool(value):
            return False
        return isinstance(value, Real)

    @staticmethod
    def _is_date_like(value: Any) -> bool:
        return isinstance(value, (date, datetime, pd.Timestamp))


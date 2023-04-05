from dataclasses import dataclass  # ! cannot use in jupyter modules
from typing import Any, Generic, TypeVar
import jsons


TP = TypeVar('TP')


@dataclass
class DataClassJsonWrapped():

    @classmethod
    def fromJson(cls, obj: Any):
        return jsons.load(obj, cls=cls)

    def toJson(self) -> dict[str, Any] | list[Any] | int | str | bool | float:
        x: Any = jsons.dump(self, strip_privates=True)
        return x

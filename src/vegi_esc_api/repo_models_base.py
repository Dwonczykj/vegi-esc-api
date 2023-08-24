from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
import jsons
from flask import Flask
from copy import copy, deepcopy
from typing import Any, Self
from typing import Callable, Protocol
from typing import TypeVar, cast
import vegi_esc_api.logger as Logger


class _DBFetchable():
    def __init__(self):
        # ...
        pass

    @abstractmethod
    def fetch(self):
        pass
    

class _DBDataClassJsonWrapped:
    @classmethod
    def fromJson(cls, obj: Any):
        return jsons.load(obj, cls=cls)

    def toJson(self) -> dict[str, Any] | list[Any] | int | str | bool | float:
        x: Any = jsons.dump(self, strip_privates=True)
        return x

    def serialize(self):
        return self.toJson()

    def __copy__(self: Self) -> Self:
        return copy(self)

    def __deepcopy__(self: Self) -> Self:
        # copy = type(self)()
        # memo[id(self)] = copy
        # copy._member1 = self._member1
        # copy._member2 = deepcopy(self._member2, memo)
        # return copy
        memo: dict[int, Any] = {}
        return deepcopy(self, memo)

    def copyWith(self: Self, **kwargs: Any) -> Self:
        c = self.__deepcopy__()
        for k, v in kwargs.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c
    

class _hasID(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.id: int


class _hasName(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.name: str
        
    
class _isCreate(_hasName):
    def __repr__(self):
        return f"{type(self).__name__}<name {self.name}>"


class _isDBInstance(_hasID, _hasName):
    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; name {self.name}>"


F = TypeVar('F', bound=Callable[..., Any])


class VegiRepoProtocol(Protocol):
    def __init__(self):
        self.app: Flask
        

# ~ https://www.notion.so/gember/Python-Cheats-8d7b0cc6f58544ef888ea36bb5879141?pvs=4#94e922507b5a426a804ea7d75574fc11
def appcontext(f: F, *args: Any, **kwargs: Any) -> F:
    if args:
        Logger.warn(f'Unknown args for appcontext decorator: {args}')
    if kwargs:
        Logger.warn(f'Unknown kwargs for appcontext decorator: {kwargs}')
    
    def deco(self: VegiRepoProtocol, *args: Any, **kwargs: Any):
        with self.app.app_context():
            try:
                return f(self, *args, **kwargs)
            except Exception as e:
                Logger.error(str(e))
                return []
    return cast(F, deco)

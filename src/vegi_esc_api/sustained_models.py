from vegi_esc_api.json_class_helpers import DataClassJsonWrapped
from dataclasses import dataclass
from typing import Generic, Literal, Any, Optional, TypeVar
import jsons


@dataclass
class SustainedProductPoints(DataClassJsonWrapped):
    pef: float


@dataclass
class SustainedCategoryLinks(DataClassJsonWrapped):
    products: str
    category: str
    next: Optional[str]


@dataclass
class SustainedProductBaseLinks(DataClassJsonWrapped):
    category: str
    impacts: str


@dataclass
class SustainedProductsLinks(SustainedProductBaseLinks):
    product: str


@dataclass
class SustainedProductLinks(SustainedProductBaseLinks):
    pass


@dataclass
class SustainedProductsListLinks(DataClassJsonWrapped):
    products: Optional[str]
    # self: Optional[str]
    next: Optional[str]
    first: Optional[str]


@dataclass
class SustainedImpactsListLinks(DataClassJsonWrapped):
    product: Optional[str]
    # self: Optional[str]
    next: Optional[str]
    first: Optional[str]


class SustainedCategoriesListLinks(DataClassJsonWrapped):
    categories: Optional[str]
    # self: Optional[str]
    next: Optional[str]
    first: Optional[str]


@dataclass
class SustainedCategory:
    id: str
    name: str
    links: SustainedCategoryLinks

    @classmethod
    def fromJson(cls, obj: Any):
        return jsons.load(obj, cls=cls)

    def toJson(self):
        return jsons.dump(self, strip_privates=True)


@dataclass
class SustainedImpact:
    id: str
    title: str
    description: str
    grade: Literal["A"] | Literal["B"] | Literal["C"] | Literal["D"] | Literal[
        "E"
    ] | Literal["F"] | Literal["G"] | str
    svg_icon: str

    @classmethod
    def fromJson(cls, obj: Any):
        return jsons.load(obj, cls=cls)

    def toJson(self):
        return jsons.dump(self, strip_privates=True)


@dataclass
class SustainedProductBase(DataClassJsonWrapped):
    id: str
    name: str
    category: str
    pack: str
    grade: Literal["A"] | Literal["B"] | Literal["C"] | Literal["D"] | Literal[
        "E"
    ] | Literal["F"] | Literal["G"] | str
    gtin: str
    image: str
    info_icons: list[str]
    points: SustainedProductPoints
    links: SustainedProductBaseLinks


@dataclass
class SustainedProductsResultItem(SustainedProductBase):
    links: SustainedProductsLinks


@dataclass
class SustainedSingleProductResult(SustainedProductBase):
    links: SustainedProductLinks


TP = TypeVar("TP", SustainedProductsResultItem, SustainedSingleProductResult)


@dataclass
class SustainedProductExplained(Generic[TP], DataClassJsonWrapped):
    product: SustainedProductBase
    impacts: list[SustainedImpact]


@dataclass
class SustainedCategoriesList(DataClassJsonWrapped):
    categories: list[SustainedCategory]
    links: SustainedCategoriesListLinks
    page: int
    page_size: int
    next_page_token: Optional[str]
    total_results: int


@dataclass
class SustainedProductsList(DataClassJsonWrapped):
    products: list[SustainedProductsResultItem]
    links: SustainedProductsListLinks
    page: int
    page_size: int
    next_page_token: Optional[str]
    total_results: int


@dataclass
class SustainedImpactsList(DataClassJsonWrapped):
    impacts: list[SustainedImpact]
    links: SustainedImpactsListLinks
    page: int
    page_size: int
    next_page_token: Optional[str]
    total_results: int



from dataclasses import dataclass
from vegi_esc_api.json_class_helpers import DataClassJsonWrapped
from vegi_esc_api.models import ESCExplanation, ESCRating
from vegi_esc_api.sustained_models import SustainedCategory, SustainedProductBase


@dataclass
class ESCRatingExplained(DataClassJsonWrapped):
    rating: ESCRating
    explanations: list[ESCExplanation]


@dataclass
class ESCRatingExplainedResult(ESCRatingExplained):
    original_search_term: str
    wmdistance: float
    _sustainedProduct: SustainedProductBase


@dataclass
class CachedSustainedItemCategory(SustainedCategory):
    products: list[SustainedProductBase]

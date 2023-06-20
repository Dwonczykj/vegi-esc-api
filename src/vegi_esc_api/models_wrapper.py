

from dataclasses import dataclass
from vegi_esc_api.json_class_helpers import DataClassJsonWrapped
from vegi_esc_api.models import ESCRatingCreate, ESCExplanationCreate, ESCProductInstance
from vegi_esc_api.sustained_models import SustainedCategory, SustainedProductBase


@dataclass
class ESCRatingExplained(DataClassJsonWrapped):
    rating: ESCRatingCreate
    explanations: list[ESCExplanationCreate]


@dataclass
class ESCRatingExplainedResult(ESCRatingExplained):
    original_search_term: str
    wmdistance: float
    _sustainedProduct: ESCProductInstance


@dataclass
class CachedSustainedItemCategory(SustainedCategory):
    products: list[SustainedProductBase]

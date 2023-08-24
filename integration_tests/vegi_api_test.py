import pytest
import requests
import os
from dotenv import load_dotenv

# pytest "/Users/joey/Github_Keep/vegi-esc/vegi-esc-api/integration_tests/vegi_api_test.py"
# or
# python "/Users/joey/Github_Keep/vegi-esc/vegi-esc-api/integration_tests/vegi_api_test.py" 
# # works beacuse of the pytest.main() call at bottom

if os.environ.get('esc_service_url') is None:
    load_dotenv('.env_test')
# BASE_URL = 'http://127.0.0.1:1337'
BASE_URL = os.environ.get('esc_service_url')
print(f'Running vegi api tests against "{BASE_URL}"')


def test_vegi_users_endpoint():
    response = requests.get(f"{BASE_URL}/vegi-users")
    data = response.json()

    # Ensure the response status is 200 OK
    assert response.status_code == 200

    # Ensure the response is a list
    assert isinstance(data, list)

    # Define the expected keys based on the given object structure
    expected_keys = [
        "createdAt", "deliveryPartner", "deliveryPartnerRole", "email", "fbUid",
        "id", "imageUrl", "isSuperAdmin", "isTester", "marketingEmailContactAllowed",
        "marketingNotificationUtility", "marketingPhoneContactAllowed", "marketingPushContactAllowed",
        "name", "phoneCountryCode", "phoneNoCountry", "role", "roleConfirmedWithOwner",
        "updatedAt", "vendor", "vendorConfirmed", "vendorRole"
    ]

    # Ensure each object in the array has the expected structure
    for obj in data:
        for key in expected_keys:
            assert key in obj

        # Ensure data types of certain fields
        assert isinstance(obj['id'], int)
        assert isinstance(obj['createdAt'], int)
        assert isinstance(obj['updatedAt'], int)
        assert isinstance(obj['name'], str)
        assert isinstance(obj['email'], str)


def test_connection_endpoint():
    response = requests.get(f"{BASE_URL}/connection")
    data = response.text

    # Ensure the response status is 200 OK
    assert response.status_code == 200

    # Ensure the response is a list
    assert isinstance(data, str)

    assert data == 'Success: True'


def test_llm_view_vector_store_endpoint():
    response = requests.get(f"{BASE_URL}/llm/view-vector-store")
    data = response.json()

    # Ensure the response status is 200 OK
    assert response.status_code == 200

    # Ensure the response is a list
    assert isinstance(data, dict)

    # Define the expected keys based on the given object structure
    expected_keys = [
        "name", "metadata"
    ]

    # Ensure each object in the array has the expected structure
    obj = data
    for key in expected_keys:
        assert key in obj

    # Ensure data types of certain fields
    assert isinstance(obj['name'], str)
    assert isinstance(obj['metadata'], dict)
    

def test_llm_query_vector_store_endpoint():
    response = requests.get(f"{BASE_URL}/llm/query-vector-store", params={"query": "Hummous"})
    data = response.json()

    # Ensure the response status is 200 OK
    assert response.status_code == 200

    # Ensure the main keys exist in the response
    for key in ["distances", "documents", "embeddings", "ids", "metadatas"]:
        assert key in data

    # Check the 'documents' key structure
    assert isinstance(data['documents'], list)
    for document_list in data['documents']:
        assert isinstance(document_list, list)
        for doc in document_list:
            assert isinstance(doc, str)
        assert [d for d in document_list if d == 'Classic houmous']  # notice correct spelling even though we query with incorrect spelling

    # Check the 'ids' key structure
    assert isinstance(data['ids'], list)
    for id_list in data['ids']:
        assert isinstance(id_list, list)
        for id_item in id_list:
            assert isinstance(id_item, str)
        assert [d for d in id_list if d == 'Classichoumous']

    # Check the 'metadatas' key structure
    assert isinstance(data['metadatas'], list)
    for metadata_list in data['metadatas']:
        assert isinstance(metadata_list, list)
        for metadata in metadata_list:
            assert isinstance(metadata, dict)
            for sub_key in ["category", "isProduct", "product_esc_id", "product_name", "source_id"]:
                assert sub_key in metadata

    # Since "distances" and "embeddings" are mentioned to be null, we can directly check their value
    assert data['distances'] is None
    assert data['embeddings'] is None
    
    
def test_rate_vegi_product_endpoint():
    response = requests.get(f"{BASE_URL}/rate-vegi-product/2")
    data = response.json()

    # Ensure the response status is 200 OK
    assert response.status_code == 200
    
    # Top level keys
    for key in ["category", "most_similar_esc_product", "new_rating", "product"]:
        assert key in data

    # Check for nested keys and types
    assert isinstance(data["category"], dict)
    for key in ["categoryGroup", "createdAt", "id", "imageUrl", "name", "updatedAt", "vendor"]:
        assert key in data["category"]

    assert isinstance(data["most_similar_esc_product"], dict)
    for key in ["brandName", "category", "dateOfBirth", "description", "id", "imageUrl", "ingredients", "keyWords", "name", "origin", "packagingType", "productBarCode", "product_external_id_on_source", "sizeInnerUnitType", "sizeInnerUnitValue", "source", "stockUnitsPerProduct", "supplier", "taxGroup"]:
        assert key in data["most_similar_esc_product"]

    assert isinstance(data["new_rating"], dict)
    for key in ["explanations", "rating"]:
        assert key in data["new_rating"]
        
    for explanation in data["new_rating"]["explanations"]:
        for key in ["evidence", "id", "measure", "rating", "reasons", "source", "title"]:
            assert key in explanation
    
    assert isinstance(data["product"], dict)
    for key in ["basePrice", "brandName", "category", "createdAt", "description", "id", "imageUrl", "ingredients", "isAvailable", "isFeatured", "name", "priority", "productBarCode", "proxyForVegiProduct", "shortDescription", "sizeInnerUnitType", "sizeInnerUnitValue", "status", "stockCount", "stockUnitsPerProduct", "supplier", "taxGroup", "updatedAt", "vendor", "vendorInternalId"]:
        assert key in data["product"]


# To run the tests, use the command `pytest <filename>.py`
if __name__ == '__main__':
    pytest.main()

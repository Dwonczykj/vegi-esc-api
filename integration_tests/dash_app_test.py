from typing import Callable
import pytest
import requests
import os
from dotenv import load_dotenv
import re
from bs4 import BeautifulSoup

# pytest "/Users/joey/Github_Keep/vegi-esc/vegi-esc-api/integration_tests/dash_app_test.py"
# or
# python "/Users/joey/Github_Keep/vegi-esc/vegi-esc-api/integration_tests/dash_app_test.py" # works beacuse of the pytest.main() call at bottom

if os.environ.get('esc_service_url') is None:
    load_dotenv('.env_test')
# BASE_URL = 'http://127.0.0.1:1337'
BASE_URL = os.environ.get('esc_service_url')
print(f'Running vegi api tests against "{BASE_URL}"')


def test_dashboard_not_logged_in_endpoint():
    response = requests.get(f"{BASE_URL}/dashboard")
    # response = requests.get(f"{BASE_URL}/login/?next=%2Fdashboard%2F")
    
    # Ensure the response status is 200 OK
    assert response.status_code == 200
    
    page_html = str(response.content)
    assert page_html is not None
    print(page_html)
    
    assert '<input id="username" name="username" required size="32" type="text" value="">' in page_html
    assert '<input id="password" name="password" required size="32" type="password" value="">' in page_html


# def test_dashboard_logged_in_endpoint():
#     initial_login_form = requests.get(f"{BASE_URL}/login/?next=%2Fdashboard%2F")
#     soup = BeautifulSoup(initial_login_form.text)
#     csrf_token = soup.find_all(attrs={"name" : "csrf_token"})[0]['value']
#     if os.environ.get('vegi_service_username') is None:
#         load_dotenv('.env_test')
#     username = os.environ.get('vegi_service_username')
#     password = os.environ.get('vegi_service_password')
#     response = requests.post(f"{BASE_URL}/login/?next=%2Fdashboard%2F", data={
#         'username': username,
#         'password': password,
#         'csrf_token': csrf_token,
#     })
    
#     # Ensure the response status is 200 OK
#     print(response.status_code)
#     assert response.status_code == 200
    
#     response = requests.get(f"{BASE_URL}/dashboard")
#     page_html = response.content
#     assert page_html is not None
#     print(page_html)
    
    
# To run the tests, use the command `pytest <filename>.py`
if __name__ == '__main__':
    # test_dashboard_logged_in_endpoint()
    pytest.main()

import requests
from bs4 import BeautifulSoup
from collections import Counter
from nltk.stem import WordNetLemmatizer

# List of strings to search for
search_strings = ["grocery", "consumer product", "recipe"]

# Lemmatize the search strings
lemmatizer = WordNetLemmatizer()
lemmatized_search_strings = [lemmatizer.lemmatize(string.lower()) for string in search_strings]

# Wikipedia search URL
base_url = "https://en.wikipedia.org/w/index.php?search="

grocery_urls: dict[str, BeautifulSoup] = {}

# Perform search for each string
for string in lemmatized_search_strings:
    search_url = base_url + string.replace(" ", "+")
    print(f"Searching '{search_url}'")
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all search results
    results = soup.find_all("li", {"class": "mw-search-result"})
    article_tags = soup.find_all("a[title='View the content page [ctrl-option-c]']")
    if article_tags:
        print("Result was an article")
        result = soup
        title = result.find("span", {"class": "mw-page-title-main"}).text
        link = result.find("a[title='View the content page [ctrl-option-c]']")["href"]

        # Fetch page content
        page_url = f"https://en.wikipedia.org/{link}"
        page_response = requests.get(page_url)
        page_soup = BeautifulSoup(page_response.content, "html.parser")

        # Count occurrences of the search string in page content
        content = page_soup.get_text().lower()
        count = content.count(string)

        # Check if the count is 5 or more
        if count >= 5:
            grocery_urls[page_url] = page_soup
            print("Title:", title)
            print("URL:", page_url)
            print("Occurrences:", count)
            print("---")
    elif results:
        print(f"Found {len(results)} results in search")
        # Check occurrence count in each search result
        for result in results:
            title = result.find("div", {"class": "mw-search-result-heading"}).find("a").text
            link = result.find("div", {"class": "mw-search-result-heading"}).find("a")["href"]

            # Fetch page content
            page_url = "https://en.wikipedia.org" + link
            page_response = requests.get(page_url)
            page_soup = BeautifulSoup(page_response.content, "html.parser")

            # Count occurrences of the search string in page content
            content = page_soup.get_text().lower()
            count = content.count(string)

            # Check if the count is 5 or more
            if count >= 5:
                grocery_urls[page_url] = page_soup
                print("Title:", title)
                print("URL:", page_url)
                print("Occurrences:", count)
                print("---")
    else:
        print(soup.get_text())

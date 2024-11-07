import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set up Chrome options for headless mode
options = Options()
options.add_argument("--headless=new")

# Initialize Selenium WebDriver in headless mode
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Open the Google Patents search page
query = input("What would you like to search for? ")
concatenating = query.replace(" ", "+")
url = f"https://patents.google.com/?q=({concatenating})"
driver.get(url)

# Allow some time for the page to load completely
time.sleep(3)

# Find the search result items
patent_results = []
results = driver.find_elements(By.CSS_SELECTOR, 'search-result-item')
print(results)

for el in results:
    # Extract the title (inside <h3> -> <span>)

    title = el.find_element(By.CSS_SELECTOR, 'h3 span#htmlContent').text.strip()

    # Link and data-result are both returning for the page as a whole rather than for particular results,
    # Which does not make sense because the other things are working correctly.
    # Extract the link (inside the <a> tag)
    link = el.find_element(By.CSS_SELECTOR, 'a#link').get_attribute('href')
    print(link)

    #link = "https://patents.google.com" + relative_link
    metadata = ' '.join(el.find_element(By.CSS_SELECTOR, 'h4.metadata').text.split())
    # Extract the data-result attribute
    data_result = el.get_attribute('data-result')
    date = el.find_element(By.CSS_SELECTOR, 'h4.dates').text.strip()
    snippet = el.find_element(By.CSS_SELECTOR, 'span#htmlContent').text

    patent_results.append({
        'title': title,
        'link': link,
        'metadata': metadata,
        'date': date,
        'snippet': snippet
    })

# Print the results as JSON
print(json.dumps(patent_results, indent=2))

# Close the WebDriver
driver.quit()
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import pandas as pd
import numpy

# Set up Chrome options for headless mode
#options = Options()
#options.add_argument("--headless=new")

# Initialize Selenium WebDriver in headless mode
#service = Service(ChromeDriverManager().install())
#driver = webdriver.Chrome(service=service, options=options)

# Open the Google Patents search page
#query = input("What would you like to search for? ")
#concatenating = query.replace(" ", "+")
#url = f"https://patents.google.com/?q=({concatenating})"
#driver.get(url)

# Allow some time for the page to load completely
#time.sleep(3)

# Find the search result items
#patent_results = []
#results = driver.find_elements(By.CSS_SELECTOR, 'search-result-item')
#print(results)

def scraper(query):
    options = Options()
    options.add_argument("--headless=new")
    # Initialize Selenium WebDriver in headless mode
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Open the Google Patents search page
    concatenating = str(query).replace(" ", "+")
    #print(concatenating)
    url = f"https://patents.google.com/?q=({concatenating})"
    #print(url)
    driver.get(url)

    # Allow some time for the page to load completely
    time.sleep(1)

    # Find the search result items
    patent_results = []
    results = driver.find_elements(By.CSS_SELECTOR, 'search-result-item')
    #print("here")
    index = 1
    for el in results:
        # Extract the title (inside <h3> -> <span>)

        title = el.find_element(By.CSS_SELECTOR, 'h3 span#htmlContent').text.strip()

        # Link and data-result are both returning for the page as a whole rather than for particular results,
        # Which does not make sense because the other things are working correctly.
        # Extract the link (inside the <a> tag)
        link = el.find_element(By.CSS_SELECTOR, 'a#link.style-scope.state-modifier').get_attribute('href')
        #print(link)

        # link = "https://patents.google.com" + relative_link
        metadata = ' '.join(el.find_element(By.CSS_SELECTOR, 'h4.metadata').text.split())
        # Extract the data-result attribute
        data_result = el.get_attribute('data-result')
        date = el.find_element(By.CSS_SELECTOR, 'h4.dates').text.split('•')[0].strip()
        snippet = el.find_element(By.CSS_SELECTOR, 'span#htmlContent').text

        number = find_string_with_numerals(metadata)

        link = "https://patents.google.com/patent/" + number + "/en"
        #link = quote(link_text)
        #encoded_url = quote(url)
        #print(link2)

        patent_results.append({
            "Original_Rank": index,
            "Name": title,
            "Link": link,
            "Number": number,
            "Date": date
        })
    #print(patent_results)
    # create header
    #head = ["Name", "Link", "Number", "Date",]

    # display table
    # print(tabulate(patent_results, headers=head, tablefmt="grid"))

    # Print the results as JSON
    #print(json.dumps(patent_results, indent=2))
    df = pd.DataFrame(patent_results, columns = ["Original Ranking", "Name", "Link", "Number", "Date"])
    df.style.format({'Link': make_clickable})
    index+=1
    return df
    # Close the WebDriver
    driver.quit()

def find_string_with_numerals(input_string):
    words = input_string.split(' ')
    for word in words:
        if re.search(r'\d{5,}', word):
            return word
    return None

def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)

# Print the results as JSON
#print(json.dumps(patent_results, indent=2))

# Close the WebDriver
#driver.quit()
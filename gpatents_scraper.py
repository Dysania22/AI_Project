import serpapi
from dotenv import load_dotenv
import pandas as pd
import os

# Load environment variables from the .env file
load_dotenv()
apik = "8144e3eea1fe3a2f04df9d5d28edeb3d7562565fe3cbf8d93f4210aa2b2948df"
# Your API Key can be found at https://serpapi.com/manage-api-key
client = serpapi.Client(api_key=apik)

all_extracted_data = [] # Master data list used to create .csv file later on
page_number = 1         # Assign initial page_number value as 1

while True:
    results = client.search({
            "engine": "google_patents", # Define engine
            "q": "(Coffee)",            # Your search query
            "page": page_number         # Page number, defined before
    })

    organic_results = results["organic_results"]

    # Extract data from each result
    extracted_data = []
    for result in organic_results:
        data = {
            "title": result.get("title"),
            "filing_date": result.get("filing_date"),
            "patent_id": result.get("patent_id")
        }
        extracted_data.append(data)

    # Add the extracted data to the master data list
    all_extracted_data.extend(extracted_data)
    # Increment page number value by 1 or end the loop
    if page_number < 2:
        page_number +=1
    else:
        break

csv_file = "extracted_data.csv" # Assign .csv file name to a variable
csv_columns = [                 # Define list of columns for your .csv file
    "title",
    "filing_date",
    "patent_id"
    ]

# Save all extracted data to a CSV file
pd.DataFrame(data=all_extracted_data).to_csv(
    csv_file,
    columns=csv_columns,
    encoding="utf-8",
    index=False
    )
print("here")
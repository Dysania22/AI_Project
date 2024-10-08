import requests
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import json
from serpapi import GoogleSearch

"""
params = {
  "engine": "google_patents",
  "q": "(Coffee)",
  "api_key": "8144e3eea1fe3a2f04df9d5d28edeb3d7562565fe3cbf8d93f4210aa2b2948df"
}

search = GoogleSearch(params)
results = search.get_dict()
print(results)
"""

# Load spaCy model for advanced preprocessing
nlp = spacy.load('en_core_web_sm')

# Initialize the SciBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

API_KEY = "vAAvEdRZ.JMceyc7bQU6WnoXFF6M79GPKz3I5Jo4d"
# API base URL for the PatentSearch API
PATENTSEARCH_API_URL = "https://search.patentsview.org/api/v1/patent/?q={'patent_date':'2007-01-09'}"


# Helper function for advanced text preprocessing with spaCy
def preprocess_text_spacy(text):
    doc = nlp(text.lower())  # Process text with spaCy
    tokens = []

    for token in doc:
        # Remove stopwords, punctuation, and keep only nouns/adjectives/verbs
        if not token.is_stop and not token.is_punct and token.pos_ in ('NOUN', 'VERB', 'ADJ'):
            tokens.append(token.lemma_)  # Lemmatize the token (e.g., "running" -> "run")

    return ' '.join(tokens)


# Function to query the PatentSearch API
def query_patentsearch(query_text, num_results=10):
    headers = {
        'X-Api-Key': API_KEY,
        "Content-Type": "application/json"
    }
    # Construct the query object
    query = {
        "patent_abstract": {"_text_any": query_text}
    }

    # API parameters
    params = {
        "q": json.dumps(query),  # Query must be in JSON format and URL encoded
        "per_page": num_results,  # Number of results to return
        "page": 1  # Page number (for pagination)
    }

    # Send GET request to the API
    response = requests.get(PATENTSEARCH_API_URL, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json().get('patents', [])
    else:
        print(f"Error querying PatentSearch API: {response.status_code}")
        return []


# Function to generate SciBERT embeddings
def get_embedding(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the mean of the token embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.numpy()


# Function to calculate cosine similarities
def calculate_similarity(query_embedding, patent_embeddings):
    similarities = cosine_similarity(query_embedding, patent_embeddings)
    return similarities


# Function to rank patents based on similarity
def rank_patents(query_text, patents):
    # Preprocess the patent abstracts using spaCy
    patent_texts = [preprocess_text_spacy(patent['patent_abstract']) for patent in patents]

    # Generate embeddings for the query and patents
    query_embedding = get_embedding(preprocess_text_spacy(query_text))
    patent_embeddings = np.vstack([get_embedding(text) for text in patent_texts])

    # Calculate similarity scores
    similarities = calculate_similarity(query_embedding, patent_embeddings)

    # Rank patents by similarity
    ranked_patents = sorted(
        zip(patents, similarities[0]), key=lambda x: x[1], reverse=True
    )

    return ranked_patents, patent_embeddings


# Function to perform K-Means clustering
def cluster_patents(patent_embeddings, num_clusters=3):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(patent_embeddings)

    return clusters, kmeans


# Main function to run the prior art search and clustering
def prior_art_search(input_patent_text, num_clusters=3):
    # Step 1: Query the PatentSearch API to get relevant patents
    print("Querying PatentSearch API for similar patents...")
    patents = query_patentsearch(input_patent_text, num_results=10)

    if not patents:
        print("No patents found.")
        return

    # Step 2: Rank patents based on their similarity to the input text
    print("Ranking patents based on semantic similarity...")
    ranked_patents, patent_embeddings = rank_patents(input_patent_text, patents)

    # Step 3: Display the top ranked patents
    print("\nTop 5 most similar patents:")
    for idx, (patent, similarity) in enumerate(ranked_patents[:5], 1):
        print(f"{idx}. Patent Number: {patent['patent_number']}")
        print(f"   Title: {patent['patent_title']}")
        print(f"   Abstract: {patent['patent_abstract']}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Date: {patent['patent_date']}")
        print("\n")

    # Step 4: Perform K-Means clustering on the patent embeddings
    print(f"Clustering patents into {num_clusters} clusters...")
    clusters, kmeans = cluster_patents(patent_embeddings, num_clusters)

    # Step 5: Display the patents in each cluster
    print("\nPatents clustered into groups:")
    for cluster_id in range(num_clusters):
        print(f"\nCluster {cluster_id + 1}:")
        for i, patent in enumerate(patents):
            if clusters[i] == cluster_id:
                print(f" - {patent['patent_number']}: {patent['patent_title']}")

    # Optional: Return the ranked patents and clusters for further use
    return ranked_patents, clusters


# Example usage
if __name__ == "__main__":
    # Input patent text (abstract/claims)
    input_patent_text = """
    A method and system for managing a distributed database system, wherein the distributed database system includes a master database and a plurality of replica databases...
    """  # Replace with actual patent text or abstract

    prior_art_search(input_patent_text, num_clusters=3)
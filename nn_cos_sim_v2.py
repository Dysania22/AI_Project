import os
import logging
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import re
from prettytable import PrettyTable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to extract fields from text files
def extract_fields_from_text(file_content):
    """Extract fields from the given text content."""
    fields = {}

    # Extract the decision cite, name, year, and opinion
    fields['decision_cite'] = re.search(r'::decision_cite::\s*(.+)', file_content).group(1).strip()
    fields['case_name'] = re.search(r'::decision_name::\s*(.+)', file_content).group(1).strip()
    fields['decision_year'] = re.search(r'::decision_year::\s*(.+)', file_content).group(1).strip()
    fields['text'] = re.search(r'::opinion::\s*(.+)', file_content, re.DOTALL).group(1).strip()

    return fields


# Load and preprocess data from a folder of text files
def load_and_preprocess_data_from_folder(folder_path):
    """Load data from text files in a folder and preprocess the text."""
    data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                fields = extract_fields_from_text(file_content)
                data.append(fields)

    logger.info(f"Loaded and processed {len(data)} text files from {folder_path}.")
    return data


# Preprocess text (tokenization and lowercasing)
def preprocess_text(text):
    """Basic text preprocessing (lowercasing and simple tokenization)."""
    return gensim.utils.simple_preprocess(text)


# Train Doc2Vec model
def train_doc2vec_model(data, vector_size=100, epochs=40):
    """Train a Doc2Vec model on the court cases."""
    # Tag each document with a unique ID (using `TaggedDocument`)
    tagged_data = [TaggedDocument(words=preprocess_text(doc['text']), tags=[i]) for i, doc in enumerate(data)]

    # Define the Doc2Vec model
    model = Doc2Vec(vector_size=vector_size, window=5, min_count=2, workers=4, epochs=epochs)

    # Build the vocabulary
    model.build_vocab(tagged_data)

    # Train the model
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    logger.info(f"Trained Doc2Vec model with {vector_size}-dimensional vectors.")

    return model


# Generate vectors for all documents
def get_doc_vectors(model, data):
    """Generate document vectors for all court cases."""
    doc_vectors = np.array([model.infer_vector(preprocess_text(doc['text'])) for doc in data])
    logger.info(f"Generated document vectors for {len(doc_vectors)} cases.")
    return doc_vectors


# Compute cosine similarities
def compute_cosine_similarities(doc_vectors):
    """Compute cosine similarity between all document vectors."""
    similarities = cosine_similarity(doc_vectors)
    logger.info(f"Computed cosine similarity matrix.")
    return similarities


# Plot the cosine similarity matrix
def plot_cosine_similarity(similarity_matrix, labels, subset_size=5):
    """Plot the cosine similarity matrix using Seaborn's heatmap."""
    if subset_size < len(similarity_matrix):
        indices = np.random.choice(range(len(similarity_matrix)), size=subset_size, replace=False)
        subset_matrix = similarity_matrix[np.ix_(indices, indices)]
        subset_labels = [labels[i] for i in indices]
    else:
        subset_matrix = similarity_matrix
        subset_labels = labels

    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Create a heatmap with annotations and a color bar
    ax = sns.heatmap(subset_matrix, cmap="coolwarm", annot=False, xticklabels=subset_labels, yticklabels=subset_labels,
                     cbar_kws={'label': 'Cosine Similarity'}, linewidths=.5)

    # Title and labels
    plt.title("Cosine Similarity Between Court Cases", fontsize=18)
    plt.xlabel("Cases", fontsize=12)
    plt.ylabel("Cases", fontsize=12)

    # Rotate the tick labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# Find the top 10 closest cases for a given case name and display in a pretty table
def find_closest_cases_for_input(case_name, cosine_sim_matrix, data, top_n=10):
    """Find the top 10 closest cases based on user input case name and display in a pretty table."""

    case_names = [doc['case_name'] for doc in data]  # Get the list of case names

    # Check if the case name exists
    if case_name not in case_names:
        print(f"Case '{case_name}' not found in the dataset.")
        return

    # Find the index of the input case
    case_index = case_names.index(case_name)

    # Get the cosine similarity scores for the input case
    similarities = cosine_sim_matrix[case_index]

    # Get the indices of the top 10 most similar cases (excluding the input case itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]

    # Get the top 10 most similar case names and their similarity scores
    similar_cases = [(case_names[i], similarities[i]) for i in similar_indices]

    # Create a PrettyTable
    table = PrettyTable()

    # Define columns
    table.field_names = ["Similar Case", "Cosine Similarity"]

    # Add the similar cases and their cosine similarity values to the table
    for similar_case, similarity in similar_cases:
        table.add_row([similar_case, f"{similarity:.4f}"])

    # Set table alignment
    table.align["Similar Case"] = "l"
    table.align["Cosine Similarity"] = "r"

    # Display the table
    print(f"\nInput Case: {case_name}")
    print(table)


# Main execution
if __name__ == "__main__":
    folder_path = 'Supreme-Court-Database/data'  # Replace with the path to your folder containing text files

    # Load and preprocess data from text files
    data = load_and_preprocess_data_from_folder(folder_path)

    # Train a Doc2Vec model on the dataset
    model = train_doc2vec_model(data, vector_size=100, epochs=5)

    # Generate vectors for each court case
    doc_vectors = get_doc_vectors(model, data)

    # Compute cosine similarities between the document vectors
    cosine_sim_matrix = compute_cosine_similarities(doc_vectors)

    # Use "case_name" field as labels for the matrix
    labels = [doc['case_name'] for doc in data]  # Use case names as labels

    # Plot the cosine similarity matrix
    plot_cosine_similarity(cosine_sim_matrix, labels)

    # Allow user input to find closest cases for a case name
    while True:
        user_input = input("\nEnter a case name to find the 10 closest cases (or type 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        # Find and display the closest cases for the input case name
        find_closest_cases_for_input(user_input, cosine_sim_matrix, data)
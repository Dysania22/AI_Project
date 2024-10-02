import pandas as pd
import logging
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load and preprocess data
def load_and_preprocess_data(file_path):
    """Loads data from CSV and preprocesses text."""
    try:
        df = pd.read_csv(file_path)
        df['date_filed'] = pd.to_datetime(df['date_filed'])
        logger.info(f"Loaded data from {file_path} with {len(df)} records.")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e


# Preprocess text (tokenization and lowercasing)
def preprocess_text(text):
    """Basic text preprocessing (lowercasing and simple tokenization)."""
    return gensim.utils.simple_preprocess(text)


# Train Doc2Vec model
def train_doc2vec_model(df, vector_size=100, epochs=40):
    """Train a Doc2Vec model on the court cases."""
    # Tag each document with a unique ID (using `TaggedDocument`)
    tagged_data = [TaggedDocument(words=preprocess_text(text), tags=[i]) for i, text in enumerate(df['text'])]

    # Define the Doc2Vec model
    model = Doc2Vec(vector_size=vector_size, window=5, min_count=2, workers=4, epochs=epochs)

    # Build the vocabulary
    model.build_vocab(tagged_data)

    # Train the model
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    logger.info(f"Trained Doc2Vec model with {vector_size}-dimensional vectors.")

    return model


# Generate vectors for all documents
def get_doc_vectors(model, df):
    """Generate document vectors for all court cases."""
    doc_vectors = np.array([model.infer_vector(preprocess_text(text)) for text in df['text']])
    logger.info(f"Generated document vectors for {len(doc_vectors)} cases.")
    return doc_vectors


# Compute cosine similarities
def compute_cosine_similarities(doc_vectors):
    """Compute cosine similarity between all document vectors."""
    similarities = cosine_similarity(doc_vectors)
    logger.info(f"Computed cosine similarity matrix.")
    return similarities


# Plot the cosine similarity matrix
def plot_cosine_similarity(similarity_matrix, labels, subset_size=15):
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


# Select 5 random cases and find 10 closest cases
def find_closest_cases(cosine_sim_matrix, df, num_cases=5, top_n=10):
    """Select 5 random cases and find their top 10 closest cases based on cosine similarity."""

    case_names = df['case_name'].tolist()  # Get the case names
    random_cases_indices = random.sample(range(len(df)), num_cases)  # Select 5 random indices

    results = []

    for idx in random_cases_indices:
        case_name = case_names[idx]
        # Get similarity scores for the selected case
        similarities = cosine_sim_matrix[idx]
        # Get the indices of the top 10 most similar cases (excluding the case itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]
        # Get the top 10 most similar case names and their similarity scores
        similar_cases = [(case_names[i], similarities[i]) for i in similar_indices]
        results.append((case_name, similar_cases))

    return results


# Display the closest cases in a readable format
def display_closest_cases(results):
    """Display the closest cases and their cosine similarity values."""

    for case_name, similar_cases in results:
        print(f"\nRandom Case: {case_name}")
        print(f"{'Similar Case':<30} {'Cosine Similarity':>18}")
        print("-" * 50)
        for similar_case, similarity in similar_cases:
            print(f"{similar_case:<30}          {similarity:>18.4f}")


# Main execution

file_path = 'all_opinions.csv'

# Load and preprocess data
df = load_and_preprocess_data(file_path)

# Train a Doc2Vec model on the dataset
model = train_doc2vec_model(df, vector_size=100, epochs=40)

# Generate vectors for each court case
doc_vectors = get_doc_vectors(model, df)

# Compute cosine similarities between the document vectors
cosine_sim_matrix = compute_cosine_similarities(doc_vectors)

# Use "case_name" column as labels for the matrix
if 'case_name' in df.columns:
    labels = df['case_name'].tolist()  # Use case names as labels
else:
    labels = [f"Case {i}" for i in range(len(df))]  # Fallback to case numbers

# Plot the cosine similarity matrix
plot_cosine_similarity(cosine_sim_matrix, labels)

# Find 5 random cases and their 10 closest cases
closest_cases = find_closest_cases(cosine_sim_matrix, df, num_cases=5, top_n=10)

# Display the closest cases and their cosine similarity values
display_closest_cases(closest_cases)
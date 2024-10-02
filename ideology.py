import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Constants
API_KEY = os.getenv("GROQ_API_KEY", "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")  # Make API key flexible
MODEL_NAME = 'gpt2'  # or 'llama-3.1-70b-versatile', use a variable for model flexibility

# Load and preprocess data
def load_and_preprocess_data(file_path):
    """Load data from CSV and preprocess dates"""
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

# Feature extraction
def extract_features(texts, max_features=5000):
    """Extract TF-IDF features from text data"""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    features = vectorizer.fit_transform(texts)
    logger.info(f"Extracted TF-IDF features with {max_features} max features.")
    return features

# Topic modeling
def perform_topic_modeling(features, n_topics=10):
    """Perform LDA-based topic modeling"""
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_matrix = lda.fit_transform(features)
    logger.info(f"Performed topic modeling with {n_topics} topics.")
    return topic_matrix

# Train generative model
def load_generative_model(model_name):
    """Load pre-trained language model and tokenizer"""
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        logger.info(f"Loaded {model_name} model and tokenizer.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

# Generate text and analyze
def generate_and_analyze(model, tokenizer, prompt, max_length=1000):
    """Generate text from a model based on a prompt and analyze it"""
    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True)
        if len(input_ids[0]) == 0:
            logger.warning("Input prompt is too short after tokenization.")
            return ""
        output = model.generate(input_ids, max_length=max_length, attention_mask=input_ids.ne(0).int(), pad_token_id=tokenizer.pad_token_id)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        return ""

# Detect ideological shifts
def detect_ideological_shifts(df, model, tokenizer):
    """Detect ideological shifts by year"""
    shifts = []
    try:
        for year in df['date_filed'].dt.year.unique():
            year_texts = df[df['date_filed'].dt.year == year]['text']
            if len(year_texts) == 0:
                logger.warning(f"No texts found for year {year}.")
                continue

            year_prompt = year_texts.iloc[0][:1024]  # Use first 1024 chars as prompt
            generated = generate_and_analyze(model, tokenizer, year_prompt)

            # Simplistic comparison for ideological shift detection
            overlap_scores = [len(set(generated.split()) & set(text.split())) for text in year_texts]
            shifts.append((year, np.mean(overlap_scores)))
            logger.info(f"Year {year}: Detected ideological shift score {np.mean(overlap_scores)}.")
    except Exception as e:
        logger.error(f"Error detecting ideological shifts: {e}")
    return shifts

# Visualization
def visualize_results(ideological_shifts):
    """Visualize ideological shifts over time"""
    df_shifts = pd.DataFrame(ideological_shifts, columns=['Year', 'Ideological Score'])
    sns.set()
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))

    # Scatter plot
    sns.scatterplot(data=df_shifts, x='Year', y='Ideological Score', alpha=0.6)

    # Trend line using Savitzky-Golay filter
    window_length = min(15, len(df_shifts) - 2)
    if window_length % 2 == 0:
        window_length -= 1
    yhat = savgol_filter(df_shifts['Ideological Score'], window_length, 3)
    #plt.plot(df_shifts['Year'], yhat, color='red', label='Trend')

    # Annotations for significant shifts
    significant_shifts = df_shifts[
        df_shifts['Ideological Score'].diff().abs() > df_shifts['Ideological Score'].diff().abs().mean() + df_shifts['Ideological Score'].diff().abs().std()]
    for _, shift in significant_shifts.iterrows():
        plt.annotate(f"{shift['Year']}",
                     (shift['Year'], shift['Ideological Score']),
                     xytext=(5, 5), textcoords='offset points')

    # Mean line
    mean_score = df_shifts['Ideological Score'].mean()
    plt.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.5)
    plt.text(df_shifts['Year'].max(), mean_score, 'Mean', verticalalignment='bottom', horizontalalignment='right')

    # Display plot
    plt.tight_layout()
    plt.show()

    # 5-year Moving Average Plot
    plt.figure(figsize=(12, 6))
    df_shifts['5 Year MA'] = df_shifts['Ideological Score'].rolling(window=5).mean()
    sns.lineplot(data=df_shifts, x='Year', y='Ideological Score', label='Yearly Score')
    sns.lineplot(data=df_shifts, x='Year', y='5 Year MA', label='5 Year Moving Average')
    plt.title('Ideological Shifts with 5 Year Moving Average', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Ideological Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = 'all_opinions.csv'
    df = load_and_preprocess_data(file_path)
    features = extract_features(df['text'])
    topics = perform_topic_modeling(features)
    model, tokenizer = load_generative_model(MODEL_NAME)
    ideological_shifts = detect_ideological_shifts(df, model, tokenizer)
    visualize_results(ideological_shifts)
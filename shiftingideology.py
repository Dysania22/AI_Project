
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

API_KEY = "YOUR_API_KEY"
model_id = "llama-3.1-70b-versatile"

file_path1 = r"C:\Users\caleb\PycharmProjects\pythonProject\all_opinions.csv"
data = pd.read_csv(file_path1)

print("here1")

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date_filed'] = pd.to_datetime(df['date_filed'])
    return df

# Feature extraction
def extract_features(texts):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    return vectorizer.fit_transform(texts)

# Topic modeling
def perform_topic_modeling(features, n_topics=10):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    return lda.fit_transform(features)

# Train generative model
def train_generative_model(texts):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Fine-tuning code here (omitted for brevity)
    return model, tokenizer

# Generate text and analyze
def generate_and_analyze(model, tokenizer, prompt, max_new_tokens=1000):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Model-specific configurations
    max_length = model.config.n_positions
    vocab_size = model.config.vocab_size
    eos_token_id = tokenizer.eos_token_id

    # Ensure input_ids do not exceed max positions
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]

    # Ensure all token ids are within the valid range
    input_ids = input_ids.clamp(0, vocab_size - 1)

    # Generate text
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        attention_mask=input_ids.ne(eos_token_id).int(),
        pad_token_id=eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# Detect ideological shifts
def detect_ideological_shifts(df, model, tokenizer):
    shifts = []
    for year in df['date_filed'].dt.year.unique():
        year_texts = df[df['date_filed'].dt.year == year]['text']
        year_prompt = year_texts.iloc[0][:1024]  # Use first 1024 chars as prompt
        generated = generate_and_analyze(model, tokenizer, year_prompt)
        # Compare generated text with actual text to detect shifts
        # This is simplified; actual implementation would be more complex
        shifts.append((year, np.mean([len(set(generated.split()) & set(text.split())) for text in year_texts])))
    return shifts

# Main execution
print("here2")
df = load_and_preprocess_data(r'C:\Users\caleb\PycharmProjects\pythonProject\all_opinions.csv')
print("here3")
features = extract_features(df['text'])
print("here4")
topics = perform_topic_modeling(features)
print("here5")
model, tokenizer = train_generative_model(df['text'])
print("here6")
ideological_shifts = detect_ideological_shifts(df, model, tokenizer)
print("here7")

# Visualization function
def visualize_results(ideological_shifts):
    df_shifts = pd.DataFrame(ideological_shifts, columns=['Year', 'Ideological Score'])
    sns.set()
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_shifts, x='Year', y='Ideological Score', alpha=0.6)

    window_length = min(15, len(df_shifts) - 2)
    if window_length % 2 == 0:
        window_length -= 1
    yhat = savgol_filter(df_shifts['Ideological Score'], window_length, 3)
    plt.plot(df_shifts['Year'], yhat, color='red', label='Trend')

    plt.title('Ideological Shifts in Supreme Court Decisions Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Ideological Score', fontsize=12)
    plt.legend(fontsize=10)

    significant_shifts = df_shifts[df_shifts['Ideological Score'].diff().abs() > (
            df_shifts['Ideological Score'].diff().abs().mean() + df_shifts['Ideological Score'].diff().abs().std())]
    for _, shift in significant_shifts.iterrows():
        plt.annotate(f"{shift['Year']}", (shift['Year'], shift['Ideological Score']), xytext=(5, 5), textcoords='offset points')

    mean_score = df_shifts['Ideological Score'].mean()
    plt.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.5)
    plt.text(df_shifts['Year'].max(), mean_score, 'Mean', verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout()
    plt.show()

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

# Call the visualization function
visualize_results(ideological_shifts)
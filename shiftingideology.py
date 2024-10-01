import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from groq import Groq
from transformers import LlamaModel, LlamaTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

API_KEY = "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"
client = Groq(api_key=API_KEY)
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
def generate_and_analyze(model, tokenizer, prompt, max_length=1000):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, attention_mask=input_ids.ne(0).int(), pad_token_id=50256)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Analysis code here
    return generated_text


# Detect ideological shifts
def detect_ideological_shifts(df, model, tokenizer):
    shifts = []
    for year in df['date_filed'].dt.year.unique():
        year_texts = df[df['date_filed'].dt.year == year]['text']
        year_prompt = year_texts.iloc[0][:100]  # Use first 100 chars as prompt
        generated = generate_and_analyze(model, tokenizer, year_prompt)
        # Compare generated text with actual text to detect shifts
        # This is a simplification; actual implementation would be more complex
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


def visualize_results(ideological_shifts):
    # Convert the shifts data to a DataFrame
    df_shifts = pd.DataFrame(ideological_shifts, columns=['Year', 'Ideological Score'])
    sns.set()
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))

    # Create the main scatter plot
    sns.scatterplot(data=df_shifts, x='Year', y='Ideological Score', alpha=0.6)

    # Add a trend line using Savitzky-Golay filter
    window_length = min(15, len(df_shifts) - 2)  # Must be odd and less than data points
    if window_length % 2 == 0:
        window_length -= 1
    yhat = savgol_filter(df_shifts['Ideological Score'], window_length, 3)
    plt.plot(df_shifts['Year'], yhat, color='red', label='Trend')

    # Customize the plot
    plt.title('Ideological Shifts in Supreme Court Decisions Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Ideological Score', fontsize=12)
    plt.legend(fontsize=10)

    # Add annotations for significant shifts
    significant_shifts = df_shifts[
        df_shifts['Ideological Score'].diff().abs() > df_shifts['Ideological Score'].diff().abs().mean() + df_shifts[
            'Ideological Score'].diff().abs().std()]
    for _, shift in significant_shifts.iterrows():
        plt.annotate(f"{shift['Year']}",
                     (shift['Year'], shift['Ideological Score']),
                     xytext=(5, 5), textcoords='offset points')

    # Add a horizontal line at the mean ideological score
    mean_score = df_shifts['Ideological Score'].mean()
    plt.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.5)
    plt.text(df_shifts['Year'].max(), mean_score, 'Mean', verticalalignment='bottom', horizontalalignment='right')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Additional analysis: Moving average plot
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

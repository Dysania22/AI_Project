# Install necessary libraries
import requests
import io
import zipfile
import pandas as pd
import random
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
# Import libraries for text preprocessing
import re  # For regular expressions
import nltk  # For text processing
nltk.download('stopwords')  # Download the list of stopwords
nltk.download('wordnet')    # Download WordNet data for lemmatization

from nltk.corpus import stopwords  # For stopwords
from nltk.stem import WordNetLemmatizer  # For lemmatization

# Import libraries for data manipulation and visualization
import pandas as pd  # For handling dataframes
import numpy as np   # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For advanced data visualization

# Import libraries for machine learning models and evaluation
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data into TF-IDF features
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For model evaluation

# Import libraries for building the Neural Network
from tensorflow.keras.models import Sequential  # For creating sequential models
from tensorflow.keras.layers import Dense  # For adding layers to the neural network
from tensorflow.keras.utils import to_categorical  # For converting labels to categorical format

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

#we start today with a quest: we have a large number of news articles, but only the ones related to politics are relevant to our defamation case
#we are going to get a large number of news articles from our client and we would need to classify them
#sure, we could read all the articles and sort them, but we are too lazy for that. Or, more politely, we find that our cclient would appreciate our billable time be spent elsewhere.

#instead, today we will train two types of models and try to evaluate which one is better
#one model will be a decision tree and the other will be based on a neural network

#our learning goals are
'''
1. understand how to preprocess models
2. figure out how to do EDA
2. understand how to evaluate models
3. understand how to read a confusion matrix
4. learn the distinction between training loss and validation loss
5. understand overfitting and how to prevent it
'''

#Fortunately, we have a dataset we can start our work from

# Download the zip file
url = "http://battleoftheforms.com/wp-content/uploads/2024/09/news-article-categories.csv.zip"
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the CSV file
csv_file = zip_file.extract("news-article-categories.csv")

# Read the CSV file
df = pd.read_csv(csv_file)

#Check out the data! Only fools blindly run code without looking at the data first and thinking about it (the technical term for looking at your data before modeling with it is "exploratory data analysis").

#first stage of an EDA is to figure out the fields in our data and what do they say

#let's see which keys we have
print(df.keys())

#let's get a feel for our data

# Select a random article
random_article = df.sample(n=1).iloc[0]
# Display the random article
print("Random Article:")
print(f"Title: {random_article['title']}")
print(f"Content: {random_article['body']}")
print(f"Category: {random_article['category']}")

# Check for missing values in each column
df.isnull().sum()

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load your dataset
# Assuming dataset is in a CSV format with columns: 'category', 'title', 'body'
# --- Fix: Fill missing values in 'body' with an empty string ---
df['body'] = df['body'].fillna('')
# Add a new column for article length (in characters)
df['article_length'] = df['body'].apply(len)

# --- Part 1: Total number of articles ---
total_articles = len(df)
print(f"Total number of articles: {total_articles}")

# --- Part 2: Number of unique categories ---
unique_categories = df['category'].nunique()
print(f"Number of unique categories: {unique_categories}")

# --- Part 3: Top 5 categories by number of articles ---
top_5_categories = df['category'].value_counts().head(5)
print("Top 5 categories by number of articles:")
print(top_5_categories)

# --- Part 4: Article with the longest title ---
longest_title = df.loc[df['title'].str.len().idxmax()]
print(f"Article with the longest title:\nTitle: {longest_title['title']}\nCategory: {longest_title['category']}\nLength: {len(longest_title['title'])} characters")

# --- Visualization 1: Bar plot showing the distribution of articles across different categories ---
plt.figure(figsize=(10,6))
sns.countplot(x='category', data=df, order=df['category'].value_counts().index)
plt.title('Distribution of Articles Across Categories')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()

# --- Visualization 2: Histogram displaying the distribution of article lengths ---
plt.figure(figsize=(10,6))
sns.histplot(df['article_length'], kde=True, bins=30)
plt.title('Distribution of Article Lengths (in characters)')
plt.xlabel('Article Length (characters)')
plt.ylabel('Frequency')
plt.tight_layout()
#plt.show()

# --- Visualization 3: Bar plot comparing the average article length for each category ---
plt.figure(figsize=(10,6))
avg_article_length = df.groupby('category')['article_length'].mean().sort_values(ascending=False)
sns.barplot(x=avg_article_length.index, y=avg_article_length.values)
plt.title('Average Article Length by Category')
plt.xticks(rotation=45)
plt.ylabel('Average Length (characters)')
plt.tight_layout()
#plt.show()

# --- Visualization 4: Word cloud visualization of the most common words in article titles ---
stopwords_set = set(stopwords.words('english'))
title_words = ' '.join(df['title'])
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set).generate(title_words)

plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Common Words in Article Titles')
plt.tight_layout()
#plt.show()

# --- Part 5: Basic summary statistics for article lengths ---
article_length_stats = df['article_length'].describe()
print("Summary statistics for article lengths:")
print(article_length_stats)

# --- Visualization 5: Box plot showing the distribution of article lengths for each category ---
plt.figure(figsize=(10,6))
sns.boxplot(x='category', y='article_length', data=df)
plt.title('Distribution of Article Lengths by Category')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()

# --- Visualization 6: Bar plot of the top 10 most frequently occurring words in article titles ---
# Tokenize the titles and remove stopwords
filtered_words = [word for title in df['title'] for word in title.lower().split() if word not in stopwords_set]
word_counts = Counter(filtered_words)
top_10_words = word_counts.most_common(10)

top_words_df = pd.DataFrame(top_10_words, columns=['word', 'count'])
plt.figure(figsize=(10,6))
sns.barplot(x='word', y='count', data=top_words_df)
plt.title('Top 10 Most Frequent Words in Article Titles')
plt.tight_layout()
#plt.show()

# the point of training is to take training examples that are labelled and have the model learn from those examples
# Look at the data, what is a label?
# what data exists other than the label?
# Get the unique categories
categories = df['category'].unique()
#print('Categories:', categories)

# Create a mapping of categories to numerical values
category_mapping = {category: idx for idx, category in enumerate(categories)}
#print('Category Mapping:', category_mapping)

# Apply the mapping to create a new column with encoded categories
df['category_encoded'] = df['category'].map(category_mapping)

# Display the encoded categories
df[['category', 'category_encoded']].head()

lemmatizer = WordNetLemmatizer()

# Define a function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    clean_text = ' '.join(words)
    return clean_text


# Apply the preprocessing function to the 'title' column
df['clean_text'] = df['title'].apply(preprocess_text)

# Display the original and cleaned text
df[['title', 'clean_text']].head()


# Assuming df, clean_text, and category_mapping are already defined from previous code

# Print available categories and their counts
#print("Available categories and their counts:")
category_counts = df['category'].value_counts()
#print(category_counts)

# Select two categories with significant number of articles
categories_to_visualize = category_counts.nlargest(2).index.tolist()

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to DataFrame for easier analysis
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Function to get top N TF-IDF scores for each category
def get_top_tfidf_scores(category, n=15):
    category_mask = df['category'] == category
    category_tfidf = tfidf_df[category_mask]

    # Calculate mean TF-IDF score for each word in the category
    mean_tfidf = category_tfidf.mean()

    # Sort and get top N words
    top_words = mean_tfidf.sort_values(ascending=False).head(n)
    return top_words

# Get top words for both categories
top_words1 = get_top_tfidf_scores(categories_to_visualize[0])
top_words2 = get_top_tfidf_scores(categories_to_visualize[1])

# Function to safely normalize and handle potential issues
def safe_normalize(series):
    max_value = series.max()
    if pd.isna(max_value) or max_value == 0:
        print(f"Warning: Unable to normalize {series.name}. All values are NaN or zero.")
        return series
    return series / max_value

# Normalize scores for better visualization
top_words1_norm = safe_normalize(top_words1)
top_words2_norm = safe_normalize(top_words2)

# Create a side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot for category1
sns.barplot(x=top_words1_norm.values, y=top_words1_norm.index, ax=ax1, palette='viridis')
ax1.set_title(f"Top 15 TF-IDF Words for '{categories_to_visualize[0]}' Category")
ax1.set_xlabel("Normalized TF-IDF Score")
ax1.set_ylabel("Words")
ax1.set_xlim(0, 1)  # Set x-axis limit

# Plot for category2
sns.barplot(x=top_words2_norm.values, y=top_words2_norm.index, ax=ax2, palette='viridis')
ax2.set_title(f"Top 15 TF-IDF Words for '{categories_to_visualize[1]}' Category")
ax2.set_xlabel("Normalized TF-IDF Score")
ax2.set_ylabel("Words")
ax2.set_xlim(0, 1)  # Set x-axis limit

plt.tight_layout()
#plt.show()

#OK, let's see how we train a model to detect the differences
#The first step is critical
#We split the data to training and validation sets

X_train, X_val, y_train, y_val = train_test_split(
    tfidf_matrix, df['category'], test_size=0.2, random_state=42, stratify=df['category']
)
#we call, as is conventional, the training data X and the labels are called y.

#we do an 80-20 split, with twenty percent of the data used for VALIDATION.

#Remember this fact for what comes next.

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Set the figure size for better readability
plt.figure(figsize=(20, 10))

# Plot the decision tree
plot_tree(dt_classifier, filled=True, feature_names=tfidf_vectorizer.get_feature_names_out(), class_names=categories, max_depth=3, fontsize=10)

# Show the plot
#plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on the validation set
y_pred = dt_classifier.predict(X_val)
print(categories)
# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
#print(f"Model Accuracy: {accuracy:.2f}")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Calculate accuracy for each category (class-specific accuracy)
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
#print("\nClass-specific Accuracy for each category:")
for i, accuracy in enumerate(class_accuracies):
    print(f"{categories[i]}: {accuracy:.2f}")


# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

# Create a mapping of categories to numerical values
category_mapping = {category: idx for idx, category in enumerate(categories)}
#print('Category Mapping:', category_mapping)

# Apply the mapping to create a new column with encoded categories
y_train_encoded = y_train.map(category_mapping)
y_val_encoded = y_val.map(category_mapping)

# Define the neural network model (Shallow)
model_1 = Sequential()

# Input layer with one hidden layer
model_1.add(Dense(70, input_dim=X_train.shape[1], activation='relu'))

# Output layer
model_1.add(Dense(len(categories), activation='softmax'))

# Compile the model
model_1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history_1 = model_1.fit(X_train.toarray(), y_train_encoded, epochs=8, batch_size=52, validation_data=(X_val.toarray(), y_val_encoded))

# Evaluate the accuracy
accuracy_1 = model_1.evaluate(X_val.toarray(), y_val_encoded, verbose=0)[1]
print(f"Model 1 (Shallow Network) Accuracy: {accuracy_1:.2f}")

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('Model 1 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('Model 1 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Predict on the validation set
y_pred_prob_nn = model_1.predict(X_val.toarray())

# Convert predicted probabilities to class labels
y_pred_nn = np.argmax(y_pred_prob_nn, axis=1)

# Classification report
report = classification_report(y_val_encoded, y_pred_nn, target_names=categories)


# Compute confusion matrix
conf_matrix_nn = confusion_matrix(y_val_encoded, y_pred_nn)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp_nn = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_nn, display_labels=categories)
disp_nn.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Confusion Matrix for Model 1')
plt.show()


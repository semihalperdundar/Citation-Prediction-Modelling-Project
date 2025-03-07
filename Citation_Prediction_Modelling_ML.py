import pandas as pd
import numpy as np
import logging
import json

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
import torch
import math

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Load training data
data = pd.DataFrame.from_records(json.load(open('train.json', 'r')))

# Fill missing values with empty strings
data.fillna('', inplace=True)

# Convert 'year' to integer format
data['year'] = pd.to_numeric(data['year'], errors='coerce').fillna(0).astype(int)

# Feature Engineering: Extract additional features from data

# Count the number of authors in the paper
data['num_authors'] = data['authors'].apply(lambda x: len(x.split(',')))

# Count the number of references cited in the paper
data['num_references'] = data['references'].apply(lambda x: len(x))

# Calculate the age of the paper by subtracting the publication year from the current year
data['paper_age'] = 2024 - data['year']

# Count the number of words in the title of the paper
data['title_word_count'] = data['title'].apply(lambda x: len(x.split()))

# Encode categorical variable 'venue' using LabelEncoder
venue_encoder = LabelEncoder()
data['venue_encoded'] = venue_encoder.fit_transform(data['venue'])

# Split dataset into training and validation sets
train_set, validation = train_test_split(data, test_size=0.15, random_state=123)

# Load test data
test = pd.DataFrame.from_records(json.load(open('test.json', 'r')))

# Fill missing values in test set
test.fillna('', inplace=True)

# Convert 'year' to integer format
test['year'] = pd.to_numeric(test['year'], errors='coerce').fillna(0).astype(int)

# Apply the same feature engineering to test set

# Count the number of authors in the test data
test['num_authors'] = test['authors'].apply(lambda x: len(x.split(',')))

# Count the number of references in the test data
test['num_references'] = test['references'].apply(lambda x: len(x))

# Calculate the age of the paper in the test data
test['paper_age'] = 2024 - test['year']

# Count the number of words in the title of the paper in the test data
test['title_word_count'] = test['title'].apply(lambda x: len(x.split()))

# Encode 'venue' in test set using previously trained encoder
venue_mapping = {venue: idx for idx, venue in enumerate(venue_encoder.classes_)}
test['venue_encoded'] = test['venue'].apply(lambda x: venue_mapping.get(x, -1))

# Apply TF-IDF transformation for text features
title_tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=3000)
abstract_tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000)

# Generate embeddings using SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

if torch.cuda.is_available():
    embedder = embedder.to('cuda')

def generate_embedding_in_batches(text_list, batch_size=128):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        batch_embeddings = embedder.encode(batch_texts, show_progress_bar=True, batch_size=batch_size, device='cuda' if torch.cuda.is_available() else 'cpu')
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Define ColumnTransformer for feature processing
featurizer = ColumnTransformer([
    ("year", 'passthrough', ['year']),
    ("num_authors", 'passthrough', ['num_authors']),
    ("num_references", 'passthrough', ['num_references']),
    ("paper_age", 'passthrough', ['paper_age']),
    ("title_word_count", 'passthrough', ['title_word_count']),
    ("venue_encoded", 'passthrough', ['venue_encoded']),
    ("title_tfidf", title_tfidf_vectorizer, 'title'),
    ("abstract_tfidf", abstract_tfidf_vectorizer, 'abstract')
], remainder='drop')

# Define ML models with their respective hyperparameters

# Ridge Regression: A linear model with L2 regularization to prevent overfitting
ridge = make_pipeline(
    featurizer, 
    Ridge(alpha=0.5, random_state=42)
)

# Gradient Boosting Regressor: An ensemble model that builds trees sequentially to minimize error
gbr = make_pipeline(
    featurizer, 
    GradientBoostingRegressor(
        n_estimators=100,   # Number of boosting stages
        learning_rate=0.01,  # Controls the contribution of each tree
        max_depth=3,        # Maximum depth of each tree
        subsample=0.5,      # Fraction of samples used for fitting individual trees
        min_samples_split=10,  # Minimum samples required to split a node
        min_samples_leaf=5,    # Minimum samples required in a leaf node
        random_state=42
    )
)

# Random Forest Regressor: A bagging ensemble of decision trees that reduces variance
rf = make_pipeline(
    featurizer, 
    RandomForestRegressor(
        n_estimators=200,   # Number of trees in the forest
        max_depth=8,        # Maximum depth of each tree
        min_samples_split=10,  # Minimum number of samples required to split an internal node
        min_samples_leaf=5,    # Minimum number of samples required to be at a leaf node
        max_features=0.8,   # Fraction of features considered for splitting
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
)

# XGBoost Regressor: An optimized gradient boosting model that handles missing data efficiently
xgb = make_pipeline(
    featurizer, 
    XGBRegressor(
        n_estimators=100,   # Number of boosting rounds
        learning_rate=0.005, # Step size shrinkage to prevent overfitting
        max_depth=3,       # Maximum depth of a tree
        gamma=1,           # Minimum loss reduction required to make a split
        subsample=0.5,     # Fraction of samples used for training
        random_state=42,
        use_label_encoder=False,  # Disable label encoding
        eval_metric='mae',  # Use Mean Absolute Error as evaluation metric
        n_jobs=-1  # Use all available CPU cores
    )
)

# Train and evaluate models
models = {"Ridge": ridge, "GradientBoosting": gbr, "RandomForest": rf, "XGBoost": xgb}
validation_scores = {}

for model_name, model in models.items():
    logging.info(f"Training {model_name}")
    model.fit(train_set.drop(columns=['n_citation'], errors='ignore'), np.log1p(train_set['n_citation'].values))
    validation_pred = np.expm1(model.predict(validation.drop(columns=['n_citation'], errors='ignore')))
    validation_mae = mean_absolute_error(validation['n_citation'], validation_pred)
    validation_scores[model_name] = validation_mae
    logging.info(f"{model_name} validation MAE: {validation_mae:.2f}")

# Identify the best model based on validation performance
best_model_name = min(validation_scores, key=validation_scores.get)
best_model = models[best_model_name]
logging.info(f"Best model is {best_model_name} with MAE: {validation_scores[best_model_name]:.2f}")

# Generate predictions for the best model
test['n_citation'] = np.expm1(best_model.predict(test))

# Save the predictions to a JSON file
json.dump(test[['n_citation']].to_dict(orient='records'), open(f'predicted_{best_model_name}.json', 'w'),indent=2)
logging.info(f"Predictions for the best model '{best_model_name}' saved to 'predicted_{best_model_name}.json'")

# Set logging level
logging.getLogger().setLevel(logging.INFO)


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv("complete_food_symptoms_dataset.csv")  # Adjust the path if needed
X = data['Food_Name']
y = data['Ingredients']

# Vectorize the food names
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier and fit it to the training data
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Save the model and vectorizer to files
joblib.dump(classifier, 'food_allergen_model.pkl')  # Save model
joblib.dump(vectorizer, 'food_vectorizer.pkl')  # Save vectorizer

print("Model and vectorizer saved successfully!")

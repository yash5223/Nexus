import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
import subprocess

app = Flask(__name__)

# Load the trained model and vectorizer
classifier = joblib.load('food_allergen_model.pkl')  # Load the saved model
vectorizer = joblib.load('food_vectorizer.pkl')  # Load the saved vectorizer

# Load the complete_food_symptoms_dataset.csv globally
data = pd.read_csv("complete_food_symptoms_dataset.csv")  # Load the dataset once

# Function to predict allergens based on food name
def predict_allergens(food_name):
    food_vectorized = vectorizer.transform([food_name])  # Vectorize the food name
    prediction = classifier.predict(food_vectorized)  # Predict allergens
    return prediction[0]

# Function to suggest ingredient replacements
def suggest_replacement(allergen):
    replacement_data = pd.read_csv("ingredient_replacements.csv")  # Replace with your own file
    # Check if allergen exists in replacements and provide the corresponding replacement
    replacement = replacement_data[replacement_data['Ingredient'] == allergen]['Replacement'].values
    if len(replacement) > 0:
        return replacement[0]
    else:
        return None

# Function to check if food exists in dataset ingredients
def check_food_in_ingredients(food_name):
    matched_foods = []
    for index, row in data.iterrows():
        food_ingredients = row['Ingredients'].split(', ')  # Split the food ingredients by comma
        if any(ingredient.lower() in food_name.lower() for ingredient in food_ingredients):
            matched_foods.append(row['Food_Name'])
    return matched_foods

@app.route('/scan', methods=['POST'])
def scan():
    if request.method == 'POST':
        food_name = request.form['food_name']
        
        # Run your external Python script using subprocess
        try:
            result = subprocess.check_output(['python', 'scan.py', food_name], text=True)
        except subprocess.CalledProcessError as e:
            result = f"Error occurred: {e}"

        # Render the result in the template
        return render_template('index.html', result=result)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None  # Initialize the result to store output
    replacement_suggestions = None  # Initialize to store replacement suggestions
    if request.method == 'POST':
        sample_food = request.form['food_name']  # Get food name from form input

        # Load the user data
        userdata = pd.read_csv("userdata.csv")  # Contains the allergy information
        userdata.columns = userdata.columns.str.strip()  # Strip any extra spaces in column names

        # Search for the user by name (you can modify this for dynamic user searching)
        search_name = "Alice"  # You can replace this with dynamic user input (e.g., request.form['user_name'])
        
        for index, row in userdata.iterrows():
            if row['Name'] == search_name or row['User_ID'] == 1:  # Search by name or user ID
                allergy_list = row['Allergy_List'].split(', ')  # Split the allergy list by comma
                
                # Print the allergens the user has
                result = f"User '{search_name}' has the following allergens: {', '.join(allergy_list)}"
                
                # Check if the food exists in the dataset
                if sample_food in data['Food_Name'].values:
                    food_ingredients = data[data['Food_Name'] == sample_food]['Ingredients'].values[0]
                    food_ingredients_list = food_ingredients.split(', ')  # Split the food ingredients by comma
                    allergens_found = []
                    # Check if any ingredient in the food matches the user's allergy list
                    for ingredient in food_ingredients_list:
                        if any(ingredient.lower() in allergy.lower() for allergy in allergy_list):
                            allergens_found.append(ingredient)

                    # If allergens are found, provide a warning and suggest replacements
                    if allergens_found:
                        replacement_suggestions = {}
                        for allergen in allergens_found:
                            replacement = suggest_replacement(allergen)
                            if replacement:
                                replacement_suggestions[allergen] = replacement
                        if replacement_suggestions:
                            result += f" The food '{sample_food}'."
                            result += "<br><strong>Replacement Suggestions:</strong>"
                            for allergen, replacement in replacement_suggestions.items():
                                result += f"<br>{allergen}: {replacement}"
                        else:
                            result += f" The food '{sample_food}' contains allergens: {', '.join(allergens_found)} but no replacements are available."
                    else:
                        predicted_allergen = predict_allergens(sample_food)
                        result += f" Predicted allergens for '{sample_food}': {predicted_allergen}. You are safe."
                
                # If the food is not found in the dataset, check for matches in ingredients
                else:
                    matched_ingredients = check_food_in_ingredients(sample_food)
                    if matched_ingredients:
                        result += f" The food '{sample_food}' is not found in the dataset directly, but it matches the following foods: {', '.join(matched_ingredients)}."
                    else:
                        result = f"The food '{sample_food}' is not found in the dataset, and no matches were found in the ingredients."

                break  # Stop the loop once the user is found

    return render_template('demo.html', result=result, replacement_suggestions=replacement_suggestions)  # Pass the result to the HTML template

# Function to run the external script using subprocess (you need to ensure your script exists)
def run_script():
    try:
        # Run the Python script (make sure the script is in the correct location)
        result = subprocess.run(['python', 'itemdetect.py'], capture_output=True, text=True)
        
        # Return the output of the script
        return jsonify({'result': result.stdout, 'error': result.stderr})
    except Exception as e:
        return jsonify({'error': str(e)})  # Fixed the missing parenthesis here

if __name__ == '__main__':
    app.run(debug=True)
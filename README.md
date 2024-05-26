## Music Recommendation System

- This repository contains a Python script that demonstrates how to create a music recommendation system using a Decision Tree Classifier. The system predicts the genre of music a user might like based on their age and gender.

## Dataset

- The dataset used in this project is music.csv, which contains user information (age and gender) and their preferred music genre.

## Requirements

- To run the script, you need the following Python libraries:

1: pandas
scikit-learn
You can install these libraries using pip:

```py
pip install pandas scikit-learn
```

## Script Overview

- The script performs the following steps:

- Import necessary libraries:

```py
python music_recommender.py
```

Example

- Here is an example of how the script might look:

```py
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree

# Load the dataset from CSV file
music_data = pd.read_csv("music.csv")

# Describe the data
print(music_data.describe())

# Check the shape of the data (number of rows and columns)
print(music_data.shape)

# Create input and output variables
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Export the Decision Tree
tree.export_graphviz(model,
                     out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=y.unique(),
                     label='all',
                     rounded=True,
                     filled=True)
```

## License

- This project is licensed under the MIT License - see the LICENSE file for details.

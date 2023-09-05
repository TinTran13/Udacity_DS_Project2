import sys

from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle

def load_data(database_filepath):
    """
    Load data from a SQLite database and prepare it for machine learning.

    Parameters:
    database_filepath (str): The filepath to the SQLite database.

    Returns:
    tuple: A tuple containing three elements - X (messages), y (categories), and category_names.
    - X (pandas.Series): The messages from the database.
    - y (pandas.DataFrame): The categories from the database.
    - category_names (list): The names of the categories.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names 

def tokenize(text):
    """
    Tokenize and preprocess a text string.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    list of str: A list of cleaned and lemmatized tokens from the input text.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens    


def build_model():
    """
    Build and configure a machine learning model pipeline.

    Returns:
    sklearn.model_selection.GridSearchCV: A grid search cross-validation model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # I've configured these parameters to reduce the size of the PKL file, which was previously too large (600MB) for uploading to GitHub with my previous settings.
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate a machine learning model's performance and print a classification report.

    Parameters:
    model (sklearn.model_selection.GridSearchCV): The trained machine learning model.
    X_test (pandas.Series): Test set messages.
    y_test (pandas.DataFrame): Test set categories.
    category_names (list): Names of the categories.

    Returns:
    None
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)

def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file using pickle.

    Parameters:
    model (object): The trained machine learning model to be saved.
    model_filepath (str): The filepath where the model should be saved.

    Returns:
    None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
     """
    Main function for training, evaluating, and saving a machine learning model.

    Usage:
    python train_classifier.py <database_filepath> <model_filepath>

    Parameters:
    None

    Returns:
    None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Kindly provide the file path for the disaster messages database.'\
              'as the first argument and the file path of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

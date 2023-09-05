import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and preprocess data from CSV files.

    Parameters:
    messages_filepath (str): Filepath to the CSV file containing messages.
    categories_filepath (str): Filepath to the CSV file containing categories.

    Returns:
    pandas.DataFrame: A DataFrame containing merged and cleaned data.
    """
    # load dataframe
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    # merge dataframe
    df = messages.merge(categories, on='id', how='inner')
    # create dataframe for 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    # Data preview
    row = categories.iloc[0,:]
    # Extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # Rename `categories`
    categories.columns = category_colnames
        # Translate category values into binary numbers, either 0 or 1
    for column in categories:
        # Assign each value to be the last character of the string.
        categories[column] = categories[column].str[-1]
        # Change the column from a string data type to numeric.
        categories[column] = categories[column].astype(int)
    categories.replace(2, 1, inplace=True)
    # Remove the original categories column from the df.
    df.drop('categories', axis=1, inplace = True)
    # Combine the original dataframe with the new categories dataframe.
    df = pd.concat([df, categories], axis=1)
    return df


def clean_data(df):
    """
    Clean a DataFrame by removing duplicate rows.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be cleaned.

    Returns:
    pandas.DataFrame: A DataFrame with duplicate rows removed.
    """
    # Duplicates clearn
    df = df.drop_duplicates()
    return df
    

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Data clearned saved to database!')
    
    else:
        print('Kindly furnish the filepaths for both the messages and categories files.'\
              'Use the datasets as the first and second arguments, respectively, as '\
              'As well as the file path of the database to store the cleaned data. '\
              'As the third argument.\n\nExample: python process_data.py'\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

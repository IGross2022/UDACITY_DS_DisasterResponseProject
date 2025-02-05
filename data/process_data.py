# Call the script: 
# python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories datasets."""
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """Clean dataframe by splitting categories and converting to numeric."""
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row to extract new column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames
    
    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

 
def save_data(df, database_filepath):
    """Save cleaned data into an SQLite database."""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')

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
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

# Updated July 11, 2020

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Load the data and merge the two datasets 
    
    Input: messages_filepath and categories_filepath as strings
    
    Output: df dataframe with merged content of the two datasets
    """
    df_message = pd.read_csv(messages_filepath)
    df_category = pd.read_csv(categories_filepath)
    
    # merge
    df = df_message.merge(df_category, on=['id'])
    
    print('success')
    
    return df

def clean_data(df):
    """
    Cleans the newly merged dataset by creating category columns, 
    removing an unnecessary, empty category column (child_alone), 
    removing incorrect related values, and removing duplicates
    
    Input: df dataframe with merged content of messagse and categories
    
    Output: df dataframe that is cleaned
    """
    # split categories into columns
    categories = df['categories'].str.split(';', expand=True)
    
    # create list of category names
    row1 = categories.iloc[0]
    names = row1.transform(lambda x: x[:-2]).tolist()
    
    # use these names as new column names
    categories.columns = names
    
    # convert the -1 or -0 in each entry to 1 or 0 
    for i in categories:
        # final character is the number we want to keep
        categories[i] = categories[i].transform(lambda x: x[-1:])
        
        # conversion to numeric
        categories[i] = pd.to_numeric(categories[i])
        
    # the related column has some entries with value 2, which represent
    # incorreect data and should be removed
    categories = categories[categories['related'] != 2]
    
    # we also see that child_alone is always 0, so that column can
    # be dropped as it never applies
    categories.drop('child_alone', axis=1, inplace = True)
    
    # we can now combine this dataset with the original one 
    # in the process, we can drop unneeded columns: original and categories

    df.drop('categories', axis=1, inplace = True)
    df.drop('original', axis=1, inplace=True)
    
    df_clean = pd.concat([df, categories], axis=1)
    
    # finally we can drop duplicated rows (there are 170)
    
    df_clean.drop_duplicates(inplace=True)
    
    df = df_clean
    
    return df
    


def save_data(df, database_filename):
    """
    Save the resultant clean file into SQLite database
    
    Input: df cleaned dataframe
           database_filename string filename for output
           
    Output: none
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql("Messages", engine, index=False, if_exists='replace')  


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
    

# updated July 12, 2020 to include a lemmatizer in the tokenize function

import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')


def load_data(database_filepath):
    """
    Loads data from the database
    
    Input: filepath to the SQLite database
    
    Output: x dataframe with features
    y dataframe with labels
    names with list of categories
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages',engine)
    
    # remove NaN
    df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    # dependent variable
    x = df['message']
    
    # explanatory variable
    y = df.drop(['id', 'message', 'genre'], axis = 1)
    
    # cateogry names
    names = list(y.columns.values)
    
    return x, y, names


def tokenize(text):
    """
    Tokenize the text
    
    Input: text 
    
    Output: tokenized - clean, tokenized text
    """
    # normalizing text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenizing
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokenized = []
    
    for k in tokens:
        tok = lemmatizer.lemmatize(k).lower().strip()
        tokenized.append(tok)
    
   
    return tokenized


def build_model():
    """
    Builds the pipeline
    
    Input: none
    
    Output: pipeline 
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    return pipeline


def evaluate_model(model, x_test, y_test, names):
    """
    Using a prepared model, evaulates on 4 metrics
    
    Input: model, x_test, y_text, names
    
    Output: all_metrics
    """
    y_test_predictions = model.predict(x_test)
    
    metrics = []
    
    for i, col in enumerate(names):
        
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col], y_test[:,i], average='weighted')
        metrics.append([accuracy, precision, recall, f1])
        
    metrics = np.array(metrics)
    all_metrics = pd.DataFrame(data=metrics, index=names, columns = ['Accuracy','Precision', 'Recall', 'F1 Score'])
    
    return all_metrics


def save_model(model, model_filepath):
    """
    Saves model as a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
import nltk
#it is not necessary to use nltk.download() if the data files are already present on your PC in the correct directory.
#nltk.download('punkt')  

import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report



def load_data(database_filepath):
    ''' 
    Initial data load from the database 
    
    INPUT:
    database_filepath - path of the database with data
    
    OUTPUT:
    X - DataFrame whith messages
    Y - DataFrame  with all categories used for messages analysies
    categories - List with categories names
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponseTable', engine)
    categories=['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    X = df['message']
    Y = df[categories]
    print(categories)
    return X, Y, categories


def tokenize(text):
    '''
    Split an lemmatize message text
    
    INPUT:
    text - String with message text
    
    OUTPUT:
    clean_tokens - List with lemmatized tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Create pipeline for predicting model 
    
    INPUT: None
    
    OUTPUT: Pipeline
    '''
    base_clf = RandomForestClassifier(
        class_weight='balanced', 
        random_state=42,
        n_estimators=20,        
        max_depth=None,           
        min_samples_split=2,    
        min_samples_leaf=1,     
        max_features='sqrt',    
        n_jobs=1
    )
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),    
        ('clf', MultiOutputClassifier(base_clf))
    ])
    # ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)))
    return pipeline


def scores_per_category(realdata, predictions):
    ''' 
    This function compares real data (from the testing dataset), compares them with predicted data and calculates all 
    respective scores
    
    INPUT: 
    realdata - categories data from the real test dataset 
    predictions - categories predicted by model
           
    OUTPUT: 
    df_scores - DataFrame with one row per each category with all its scores
    '''

    all_scores = []

    for col in range(realdata.shape[1]):
        # Get the true values and predicted values for the current column (category)
        true_values = realdata.iloc[:, col]
        predicted_values = predictions[:, col]
        
        # Calculate accuracy, precision, recall, and f1 score for the current category
        accuracy = accuracy_score(true_values, predicted_values)
        precision = precision_score(true_values, predicted_values, average='macro', zero_division=0)  # Controls undefined precision
        recall = recall_score(true_values, predicted_values, average='macro', zero_division=0)  # Set undefined recall to 0
        f1 = f1_score(true_values, predicted_values, average='macro', zero_division=0)  # Controls undefined F1-score

        # Append scores for this category to all_scores
        all_scores.append([realdata.columns[col], accuracy, precision, recall, f1])

    # Create a DataFrame with the scores for each category
    df_scores = pd.DataFrame(all_scores, columns=['Category', 'Accuracy', 'Precision', 'Recall', 'F1'])
    
    return df_scores


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Model testig 
    
    INPUT:
    model - pipeline
    X_test - DataFrame with test messages
    Y_test - DataFrame with categories data 
    category_names - List with names of the categories
    
    OUTPUT:
    printing out Accuracy, Precision, Recall, and F1 scores per each category
    '''
    Y_pred = model.predict(X_test)
    print(scores_per_category(Y_test,Y_pred))
    

def save_model(model, model_filepath):
    '''
    Saving the model
    
    INPUT:
    model - model to save
    model_filepath - path where to save model
    
    OUTPUT: pickle file
    
    To load the model later:
    # Load the model from the file
    with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

    # Use the loaded model
    print(loaded_model.predict(X_test))
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
        
        print('Building model...')
        model = build_model()
        print(model)

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        #print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

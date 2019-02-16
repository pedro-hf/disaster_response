import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from joblib import dump
nltk.download(['punkt', 'wordnet'])


def clean_filepath(filepath):
    """Takes a file path and it returns it without file extension, it also gives the name of the table"""
    if filepath.endswith('.db'):
        clean_path = filepath[:-3]
    else:
        clean_path = filepath
    table_name = clean_path.split('/')[-1]
    return clean_path, table_name


def load_data(database_filepath):
    """
    It loads data from a sqlite database and returns X (messages) and y (categories)
    :param database_filepath: string, path to the database file.
    :return: pd.Series, pd.DataFrame, list of strings. X is series with the messages, y is a dataframe with one-hot enconding of the
    categories. category_names is a list of the names of the categories or columns of y.
    """
    db_name, table_name = clean_filepath(database_filepath)
    engine = create_engine('sqlite:///{}.db'.format(db_name))
    df = pd.read_sql_table(table_name, con=engine)
    X = df.message
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    """
    Tokenizer to be passed to the bag of Words
    :param text: string, message to the tokenized
    :return: list of strings, list of the tokens extracted from the text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t).lower().strip() for t in tokens]


def build_model():
    """
    Creates a machine learning pipeline including: a vectorizer, a TF-IDF transformer and a classifier
    :return: returns a model
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,
                                                                                                 criterion='gini'),
                                                                          n_estimators=100)))])
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    Function that prints to terminal the performance of the model
    :param model: ml pipeline, model to evaluate.
    :param X_test: pd.DataFrame, pd.Series or np.array. Test messages.
    :param y_test: pd.DataFrame. one hot encoding dataframe with the true labels of X_test
    :param category_names: list(strings). name of the categories.
    :return: none, prints to screen
    """
    y_pred = model.predict(X_test)
    for i, cat in enumerate(category_names):
        print('######################## ' + cat + ' ########################')
        print(classification_report(y_test[cat], y_pred[:, i]))
    return None


def save_model(model, model_filepath):
    """ saves model to model_filepath"""
    dump(model, '{}.joblib'.format(model_filepath))
    return None


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
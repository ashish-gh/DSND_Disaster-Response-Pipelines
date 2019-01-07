import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection  import GridSearchCV

nltk.download(['punkt','stopwords','wordnet'])


def load_data(database_filepath):
    """
    Load datasets from local SQLite database

    Args:
    database_filename: string. Filename for SQLite database containing cleaned message data.

    Returns:
    X: dataframe. Dataframe containing features dataset.
    y: dataframe. Dataframe containing labels dataset.
    col_names: List of strings. List containing category names.
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleaned_messages', engine)

    # drop nan values
    df.dropna(axis=0, how = 'any', inplace = True)

    X = df['message']
    y = df.iloc[:,4:].astype(int)
    col_names = y.columns.values

    return X, y, col_names


def tokenize(text):
    """
    Normalize, tokenize and lemmatize text string

    Args:
    text: string. String containing message for processing

    Returns:
    clean_tokens: list of strings. List containing normalized and lemmatized word tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline

    Args:
    None

    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the
    model object and finds the optimal model parameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__min_df': [1, 5],
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators':[10, 25],
        'clf__estimator__min_samples_split':[2, 5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def get_evaluation(y_test, y_pred, category_names):
    """
    Evaluate model performance based on accuracy, precision, recall, F1 score

    Args:
    y_test: array. Actual labels.
    y_pred: array. Predicted labels.
    category_names: list of strings. List containing names for each of the predicted fields.

    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    for i in range(len(category_names)):
        accuracy = accuracy_score(y_test[:,i], y_pred[:,i])
        precision = precision_score(y_test[:,i], y_pred[:,i], average='micro')
        recall = recall_score(y_test[:,i], y_pred[:,i], average='micro')
        f1 = f1_score(y_test[:,i], y_pred[:,i], average='micro')

        metrics.append([accuracy, precision, recall, f1])

    df = pd.DataFrame(data = np.array(metrics), index=category_names, columns=['Accuracy', 'Precision', 'Recall', 'F1 score'])

    return df

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance based on accuracy, precision, recall, F1 score

    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    category_names: list of strings. List containing names for each of the predicted fields.

    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall
    and f1 score for a given set of actual and predicted labels.
    """
    Y_pred = model.predict(X_test)

    print(get_evaluation(np.array(Y_test), Y_pred, category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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

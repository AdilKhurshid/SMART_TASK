""" This script trains the training dataset on Multinomial Naive Bayes, and the
trained model is used for prediction of categories in the testing dataset.

    Returns:
        Accuracy of the trained model.
"""
from sklearn import metrics
import pandas as pd
import dataset_handler as dh
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def nb(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Trains the model on NB classifier, and predicts on the testing dataset.

    Args:
        df_train (pd.DataFrame): Dataset used for training the model.
        df_test (pd.DataFrame): Dataset used for testing the trained model.

    Returns:
        predicted_nb: The predicted values by the trained svm.
        Accuracy: Accuracy of the trained model.
    """
    text_clf = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB(alpha=1e-1)),
        ]
    )
    text_clf = text_clf.fit(df_train["question"], df_train["category_new"])
    predicted_nb = text_clf.predict(df_test["question"])
    Accuracy = metrics.accuracy_score(df_test["category_new"], predicted_nb)
    return predicted_nb, Accuracy


if __name__ == "__main__":
    df_train, df_test = dh.get_dataframe()
    predicted_nb, Accuracy = nb(df_train, df_test)
    print("Accuracy of Multinomial Naive Bayes obtained is ", Accuracy.round(2))

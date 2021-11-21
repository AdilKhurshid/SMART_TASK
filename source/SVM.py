"""
This script trains the training dataset on Support Vector Machine,
 and the trained model is used for prediction of categories in the testing dataset.

    Returns:
        Accuracy of the trained model.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
import dataset_handler as dh


def svm(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Trains the model on SVM classifier, and predicts on the testing dataset.

    Args:
        df_train (pd.DataFrame): Dataset used for training the model.
        df_test (pd.DataFrame): Dataset used for testing the trained model.

    Returns:
        predicted_svm: The predicted values by the trained svm.
        Accuracy: Accuracy of the trained model.
    """

    text_clf_svm = Pipeline(
        [("vect", TfidfVectorizer()), ("clf-svm", SVC(kernel="linear"))]
    )

    text_clf_svm = text_clf_svm.fit(df_train["question"], df_train["category_new"])
    predicted_svm = text_clf_svm.predict(df_test["question"])
    Accuracy = metrics.accuracy_score(df_test["category_new"], predicted_svm)
    return predicted_svm, Accuracy


if __name__ == "__main__":
    df_train, df_test = dh.get_dataframe()
    predicted_svm, Accuracy = svm(df_train, df_test)
    print("Accuracy of SVM obtained is ", Accuracy.round(2))

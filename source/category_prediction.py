from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os
import dataset_handler as dh
import SVM
import NB
import Roberta as rob


def load_data():
    """Load data from data set
    Returns:
        pd.DataFrame: returns train and test dataFrames .
    """
    df_train, df_test = dh.get_dataframe()
    df_train = dh.preprocess_dataframe(df_train)
    df_test = dh.separate_cat(df_test)
    return df_train, df_test


if __name__ == "__main__":
    # loading dataset
    df_train, df_test = load_data()

    # Applying SVM for Category prediction
    predicted_svm, Accuracy = SVM.svm(df_train, df_test)
    print("################# SVM ########################")
    print("Accuracy of SVM obtained is :", Accuracy.round(2))

    # Applying Naive Bayes for Category prediction
    predicted_nb, Accuracy = NB.nb(df_train, df_test)
    print("################# NB ########################")
    print("Accuracy of Multinomial Naive Bayes obtained is ", Accuracy.round(2))

    # Applying Roberta Simple Transform for Category prediction
    print("################# Roberta ########################")
    df_train_roberta = rob.value_replace(df_train)
    df_test_roberta = rob.value_replace(df_test)
    model = rob.simple_transformer(df_train_roberta[["question", "category_new"]])
    result, model_output, wrong_predictions = model.eval_model(df_test_roberta[["question", "category_new"]],
                                                               acc=sklearn.metrics.accuracy_score)
    print("################# Roberta ########################")
    print("Accuracy of Roberta obtained is:", result)

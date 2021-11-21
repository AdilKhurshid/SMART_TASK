# imports
import pandas as pd
import numpy as np
import json
import os
import dataset_handler as dh
import SVM
from gensim.summarization.bm25 import BM25


# Concatenating all the sentences of same type entity
def get_index_dict(df: pd.DataFrame) -> dict:
    """Takes a column and append all the values of another columns for same value .
    Args:
        df (pd.DataFrame): Dataset used for training the model.

    Returns:
        dict: dictionary of all types as key and all the questions of that type as value in a
        concatenated string
    """
    index = 0
    resource_index = {}
    types = list(df['type'])
    for x in types:
        for term in x:
            if term in resource_index.keys():
                resource_index[term] = resource_index[term] + ' ' + df.iloc[index]['question']
            else:
                resource_index[term] = df.iloc[index]['question']
        index = index + 1
    return resource_index


# Implementation of BM25 for Type prediction
def get_key(dic, val):
    # function to match keys and values
    for key, value in dic.items():
        if val == value:
            return key


def train_on_BM25(df_train_resource, corpus):
    # Training bm25 on training DataSet
    tok_corpus = [s.split() for s in corpus]
    bm25 = BM25(tok_corpus)
    return bm25


def get_values(bm25, train_indexs_dic, corpus, query):
    # Training bm25 on training DataSet
    # return list of top 10 relevant predicted types
    query = query.split()
    scores = bm25.get_scores(query)
    best_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: 10]
    values = []
    for s in best_docs:
        values.append(get_key(train_indexs_dic, corpus[s]))
    return values


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
    # loading DataSet
    df_train, df_test = load_data()

    # Applying SVM for Category prediction
    predicted_svm, Accuracy = SVM.svm(df_train, df_test)
    print("Accuracy of SVM obtained is :", Accuracy.round(2))

    # Creating new testing and training Dataframes for type predictions
    df_test['predicted_category'] = predicted_svm
    df_train_resource = df_train.loc[df_train['category'] == 'resource']
    df_test_resource = df_test.loc[df_test['predicted_category'] == 'resource']

    # Arranging dataset for BM25
    train_resource_index = get_index_dict(df_train_resource)
    corpus = list(train_resource_index.values())

    # training BM25 Model on training dataset
    bm25 = train_on_BM25(df_train_resource, corpus)

    # getting prediction from bm25 model and storing first 10 to relevant types
    results = {}
    for q in list(df_test_resource['question']):
        results[q] = get_values(bm25, train_resource_index, corpus, q)

    # Rearranging results according to evaluation JSON format
    final_json = dh.re_arrange_json(df=df_test, results=results)

    # write the results in prediction_output_json file for evaluation
    f = open("prediction_output_json.json", "w")
    json.dump(final_json, f)
    f.close()

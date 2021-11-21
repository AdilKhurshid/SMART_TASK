"""This script loads the dataset which is to be saved in the folder 
    "datasets/DBpedia/" and converts it into a pandas dataframe.

    Returns:
        [pd.DataFrame]: [Training and Testing dataset]
"""

import json
import os

import numpy as np
import pandas as pd


def load_dataset(path: str) -> dict:
    """Loads the dataset from the path.
    Args:
        path (str): relative path of the data files.

    Returns:
        dict: A dictionary with keys and values.
    """
    f = open(path, "r")
    data_set = json.load(f)
    f.close()
    return data_set


def load_dataset_as_dataframe(path: str) -> pd.DataFrame:
    """Loads the dataset from the path and Convert it into pandas dataframe.
    Args:
        path (str): relative path of the data files.

    Returns:
        pd.DataFrame: A pandas Datafram with keys as Columns and values as entries.
    """
    return pd.DataFrame(load_dataset(path))


def re_arrange_json(df: pd.DataFrame, results: dict) -> dict:
    """ Rearrange the results according to the evaluation JSON format .
    Args:
        df (pd.DataFrame): Dataset used for testing the model.
        results (dict): contain questions and predictions as key and values
    Returns:
        dict: dictionary of all category and types predictions
    """
    final_pre = df.to_dict('records')
    for value in final_pre:
        if value['predicted_category'] == 'number':
            value['result_category'] = 'literal'
            value["result"] = ['number']
        if value['predicted_category'] == 'string':
            value['result_category'] = 'literal'
            value["result"] = ['string']
        if value['predicted_category'] == 'date':
            value['result_category'] = 'literal'
            value["result"] = ['date']
        if value['predicted_category'] == 'boolean':
            value['result_category'] = 'boolean'
            value["result"] = ['boolean']
        if value['predicted_category'] == 'resource':
            value['result_category'] = 'resource'
            value['result'] = results[value['question']]
        del value['category']
        del value['type']
        del value['category_new']
        del value['predicted_category']

        value['category'] = value.pop('result_category')
        value['type'] = value.pop('result')
    return final_pre


def separate_cat(df: pd.DataFrame) -> pd.DataFrame:
    df["category_new"] = np.where(df["category"] == "literal", df["type"].str[0], df["category"])
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Take dataframe  arrange category in a new columns and drop nan vales  .
    Args:
        df (pd.DataFrame): Dataset used for testing the model.
    Returns:
        pd.DataFrame: with additional column category_new
    """
    df = separate_cat(df)
    df = df.dropna()
    return df


def get_dataframe():
    cur_path = os.path.dirname(os.path.abspath("__file__"))
    new_path = cur_path.replace("source", "")
    train_path = new_path + "datasets/DBpedia/smarttask_dbpedia_train.json"
    test_path = new_path + "datasets/DBpedia/smarttask_dbpedia_test.json"
    df_train = load_dataset_as_dataframe(train_path)
    df_test = load_dataset_as_dataframe(test_path)
    return df_train, df_test


if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath("__file__"))
    new_path = cur_path.replace("source", "")
    train_path = new_path + "datasets/DBpedia/smarttask_dbpedia_train.json"
    test_path = new_path + "datasets/DBpedia/smarttask_dbpedia_test.json"
    df = load_dataset_as_dataframe(train_path)
    print(df)

# imports
import pandas as pd
import numpy as np
import json
import os
import dataset_handler as dh
import SVM
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
import logging
import torch


def re_arrange_dataframe(df: pd.DataFrame):
    df_train_dict_list = df.to_dict('records')
    types_list = []
    question_list = []
    for x in df_train_dict_list:
        types_list.extend(x['type'])
        types_list = list(set(types_list))
        question_list.append(x['question'])

    train_df = pd.DataFrame(question_list, columns=['question'])
    train_df = pd.concat([train_df, pd.DataFrame(columns=types_list)])
    train_df = train_df.fillna(0)

    ls = []
    for indx, val in enumerate(df_train_dict_list):
        for x in val['type']:
            train_df.at[indx, str(x)] = 1
        temp = list(train_df.iloc[indx])
        temp.pop(0)
        ls.append(temp)

    train_df['lables'] = ls

    train_data = train_df.filter(['question', 'lables'], axis=1)
    return train_data, train_df


def simple_transformer(df: pd.DataFrame):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model_args = MultiLabelClassificationArgs(num_train_epochs=1)
    use_cuda = torch.cuda.is_available()

    # Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=306,
        args=model_args,
        use_cuda=use_cuda)

    # Train the model
    model.train_model(df)
    return model


def gen_predictions(df_test_resource, train_df, model) -> dict:
    labels_columns = list(train_df.columns)
    labels_columns.pop(0)
    labels_columns.pop(len(labels_columns) - 1)
    questions_list = df_test_resource['question'].to_list()
    predictions, raw_outputs = model.predict(questions_list)
    temp_results = dict(zip(questions_list, predictions))
    result = {}
    for key, val in temp_results.items():
        result[key] = []
        for indx, v in enumerate(val):
            if v == 1:
                result[key].append(labels_columns[indx])
    return result


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

    # Rearranging Data frame and binary to run simple transform
    train_data, train_df = re_arrange_dataframe(df=df_train_resource)

    # Training simple transform Model
    model = simple_transformer(df=train_data)

    # Generating prediction value using trained model
    result = gen_predictions(df_test_resource, train_df, model)

    # Arranging predicted values according to required evaluation file
    final_json = dh.re_arrange_json(df_test_resource, result)

    # write the results in prediction_output_json file for evaluation
    f = open("simple_transform_output_json.json", "w")
    json.dump(final_json, f)
    f.close()

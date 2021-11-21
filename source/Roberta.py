import os
import pandas as pd
import sklearn
import logging
import os
from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs,
)
import torch
import dataset_handler as dh


def value_replace(df: pd.DataFrame) -> pd.DataFrame:
    """Converts label values from string to int, as roberta base accepts only
    integers.

    Args:
        df (pd.DataFrame): Dataframe whose label values have to be changed.

    Returns:
        pd.DataFrame: Dataframe with integer label values.
    """
    df["category_new"].replace(
        {"resource": "0", "boolean": "1", "date": "2", "string": "3", "number": "4"},
        inplace=True,
    )
    df["category_new"] = df["category_new"].astype(str).astype(int)
    return df


def simple_transformer(df_train: pd.DataFrame):
    """Trains the dataset on simple transformer, roberta-base. The model args
    are given, which can be changed to fine tune the model.

    Args:
        df_train (pd.DataFrame): The dataframe to be trained on.

    Returns:
        A trained roberta base model
    """

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    model_name = "roberta-base"
    arch = "roberta"
    epochs = 6
    NUM_LABELS = 5

    model_args = ClassificationArgs(
        num_train_epochs=epochs,
        reprocess_input_data=True,
        overwrite_output_dir=True,
        save_eval_checkpoints=False,
        save_model_every_epoch=False,
        use_multiprocessing=True,
        evaluate_during_training_verbose=True,
        train_batch_size=64,
        n_gpu=2,
        logging_steps=2000,
        save_steps=20000,
        learning_rate=1e-5,
        adam_epsilon=1e-8,
        warmup_ratio=0.06,
        warmup_steps=0,
        max_grad_norm=1.0,
    )
    model = ClassificationModel(
        arch,
        model_name,
        args=model_args,
        num_labels=NUM_LABELS,
        use_cuda=torch.cuda.is_available(),
    )
    print(df_train)
    model.train_model(df_train)
    return model


if __name__ == "__main__":
    df_train, df_test = dh.get_dataframe()
    df_train = value_replace(df_train)
    df_test = value_replace(df_test)
    model = simple_transformer(df_train[["question", "category_new"]])
    result, model_output, wrong_predictions = model.eval_model(
        df_test[["question", "category_new"]], acc=sklearn.metrics.accuracy_score
    )
    print("Results:", result)

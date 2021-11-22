# SMART_TASK

## Task Description & Dataset 
For complete details about task and to download the dataset click [here](https://smart-task.github.io/2020/)

## Steps to Run File:
Clone the repository and add source file in the project folder like `/SMART_TASK/source`

## For Category Prediction:
- Run `category_prediction.py` file for category prediction. It will print the accuracy of all three models (SVM, Simple Transformer, Naive Bayes)on console.

## For Type Prediction:
- Run `BM25_with_clustering_model.py` for clustering model. It will create `clustering_output_json.json` with predicted values which further can be use for evaluation.  
- Run `simple_transformer_model.py` for simple transformer model. It will create `simple_transform_output_json.json` for evaluation.
- Run `type_centric_model.py` for type centric model. It will produce `prediction_output_json.json` for evaluation.

## Evaluation 
To evaluate the model run `evaluate.py` with generated JSON files according to the description in `smart-dataset/evaluation/dbpedia/evaluate.py`. 

### Note:
All the model are trained and evaluated only for dbpedia dataset.
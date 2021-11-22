# imports
import pandas as pd
import dataset_handler as dh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from gensim.summarization.bm25 import BM25
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import SVM


def load_data():
    """Load data from data set
    Returns:
        pd.DataFrame: returns train and test dataFrames .
    """
    df_train, df_test = dh.get_dataframe()
    df_train = dh.preprocess_dataframe(df_train)
    df_test = dh.separate_cat(df_test)
    return df_train, df_test


def tf_idf(df: pd.DataFrame):
    """Calculates the tf_idf score for all the questions in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe which contains a column "question".

    Returns:
        text: tf_idf scores.
    """
    tfidf = TfidfVectorizer(
        min_df=5, max_df=0.95, max_features=8000, stop_words="english"
    )
    tfidf.fit(df.question)
    text = tfidf.transform(df.question)
    return text


def find_optimal_clusters(data, max_k):
    """To find the optimal number of clusters to be created.

    Args:
        data ([type]): Data on which clusters have to be created.
        max_k ([type]): Max number of clusters to check.
    """
    iters = range(2, max_k + 1, 2)
    sse = []
    for k in iters:
        sse.append(
            MiniBatchKMeans(
                n_clusters=k, init_size=1024, batch_size=2048, random_state=20
            )
                .fit(data)
                .inertia_
        )
        print("Fit {} clusters".format(k))

    f, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.plot(iters, sse, marker="o")
    ax.set_xlabel("Cluster Centers")
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel("SSE")
    ax.set_title("SSE by Cluster Center Plot")


def K_mean(text: str, df: pd.DataFrame):
    """Creates clusters using K-means.

    Args:
        df (pd.DataFrame): Dataframe which contains a column "type".

    Returns:
        df: A column created with a name clusters in the input dataframe is
        returned.
    """
    kmeans = KMeans(n_clusters=64, n_init=3, max_iter=3000, random_state=1)
    kmeans = kmeans.fit(text, df_train["type"])
    df.loc[:, "clusters"] = kmeans.labels_
    return df


def svm(df_train: pd.DataFrame, df_test_resource: pd.DataFrame):
    """Predicts the cluster numbers using the trained model.

    Args:
        df_train (pd.DataFrame): Training dataset.

    Returns:
        predicted_svm: predictions of the model.
    """

    text_clf_svm = Pipeline(
        [("vect", TfidfVectorizer()), ("clf-svm", SVC(kernel="linear"))]
    )

    text_clf_svm = text_clf_svm.fit(df_train["question"], df_train["clusters"])
    predicted_svm = text_clf_svm.predict(df_test_resource["question"])
    return predicted_svm


def get_index_dict(df: pd.DataFrame):
    """Concatenating all the sentences of same type entity.

    Args:
        df ([pd.DataFrame]): Dataframe which contains questions to be
        concatenated.

    Returns:
        resource_index: dictionary which contains key as answer type and value
        as all the concatenated question of that answer type.
    """
    index = 0
    resource_index = {}
    types = list(df["type"])
    for x in types:
        for term in x:
            if term in resource_index.keys():
                resource_index[term] = (
                        resource_index[term] + " " + df.iloc[index]["question"]
                )
            else:
                resource_index[term] = df.iloc[index]["question"]
        index = index + 1
    return resource_index


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
    best_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    values = []
    for s in best_docs:
        values.append(get_key(train_indexs_dic, corpus[s]))
    return values


def get_results(df_test_resource: pd.DataFrame):
    """Ranks all the documents within the predicted cluster.

    Args:
        df_test_resource ([pd.DataFrame]): Testing dataframe which contains the
        questions and predicted cluster.

    Returns:
        results: Dictionary with keys as questions and values as top 10 ranked
        answer types.
    """
    results = {}
    for index, row in df_test_resource.iterrows():
        query = row["question"]
        cluster_number = row["predicted_cluster"]
        cluster_df = df_test_resource.loc[
            df_test_resource["predicted_cluster"] == cluster_number
            ]
        train_indexs_dic = get_index_dict(df=cluster_df)
        corpus = list(train_indexs_dic.values())
        bm25 = train_on_BM25(cluster_df, corpus)
        results[query] = get_values(bm25, train_indexs_dic, corpus, query)
    return results


if __name__ == "__main__":
    # loading DataSet
    df_train, df_test = load_data()

    # Applying SVM for Category prediction
    predicted_svm, Accuracy = SVM.svm(df_train, df_test)
    print("Accuracy of SVM obtained is :", Accuracy.round(2))

    # Creating new testing and training Dataframes for type predictions
    df_test["predicted_category"] = predicted_svm
    df_train_resource = df_train.loc[df_train["category"] == "resource"]
    df_test_resource = df_test.loc[df_test["predicted_category"] == "resource"]

    # Calculating Tf-idf and creating clusters using kmeans
    text = tf_idf(df_train_resource)
    # find_optimal_clusters(text, 100) #The optimal result found was 64
    df_train_resource = K_mean(df_train_resource, text)

    # Training and predicting clusters
    predicted_svm = svm(df_train_resource, df_test_resource)
    df_test_resource["predicted_cluster"] = predicted_svm

    # Ranking using BM25 to get top 10 answer types
    results = get_results(df_test_resource)
    final_json = dh.re_arrange_json(df=df_test, results=results)

    # Saving the predictions as json
    f = open("clustering_output_json.json", "w")
    json.dump(final_json, f)
    f.close()

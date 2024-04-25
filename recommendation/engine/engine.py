import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json
import os
from collections import Counter
import os
from numpy.linalg import norm
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import warnings
import random

warnings.filterwarnings(action="ignore")


stopwordlist = [
    "a",
    "about",
    "above",
    "after",
    "again",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "by",
    "can",
    "d",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "now",
    "o",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "own",
    "re",
    "s",
    "same",
    "she",
    "shes",
    "should",
    "shouldve",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "thatll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "y",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
]


# def download_nltk_package(package_name, subfolder):
#     try:
#         nltk.data.find(f"{subfolder}/{package_name}")
#         print("Already downloaded", package_name)
#     except LookupError:
#         print("Downloading", package_name)
#         nltk.download(package_name)


def download_nltk_package(package_name, subfolder):
    nltk_data_path = nltk.data.path[
        0
    ]  # Usually the first path is the nltk_data directory
    package_path_folder = os.path.join(nltk_data_path, subfolder, package_name)
    package_path_zip = os.path.join(nltk_data_path, subfolder, f"{package_name}.zip")

    if not (os.path.exists(package_path_folder) or os.path.exists(package_path_zip)):
        print(f"Downloading {package_name}")
        nltk.download(package_name)
    else:
        print(f"Path already exists, therefore already downloaded {package_name}")


# Use the function to download the packages conditionally
download_nltk_package("punkt", "tokenizers")
download_nltk_package("wordnet", "corpora")
download_nltk_package("omw-1.4", "corpora")

# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")


def preprocessing(corpus):
    # Replaces escape character with space
    corpus = corpus.replace("\n", " ")
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    # replace non-alphabetic characters with space
    alphaPattern = "[^a-zA-Z0-9]"

    data = []
    # iterate through each sentence in the file
    for i in sent_tokenize(corpus):
        words = []
        # tokenize the sentence into words
        for word in word_tokenize(i):
            if len(word) > 1 and word not in stopwordlist:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                words.append(re.sub(alphaPattern, " ", word.lower()))
        data.append(words)
    return data


def similarity_sentence(model, sent1, sent2):
    d1 = preprocessing(sent1)
    d2 = preprocessing(sent2)
    # flatten the d1,d2 list
    d1 = [item for sublist in d1 for item in sublist]
    d2 = [item for sublist in d2 for item in sublist]
    acc = 0
    for w1 in d1:
        for w2 in d2:
            acc += model.wv.similarity(w1, w2)
    return acc / (len(d1) * len(d2))


def _weighted_rating(x, m, C):
    v = x["votes_count"]
    R = x["vote_average"]
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


def _dataframe_related(df, category):
    # check if the category is in the dataframe
    if category not in df["category"].unique():
        pass
    else:
        df = df.loc[df["category"] == category]

    return df


def _get_similar_title(title, num_recommendations, cosine_sim, indices, metadata):
    # Get the index of the library that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all library with that index
    sim_scores = list(enumerate(cosine_sim[idx]))

    metadata["scores"] = cosine_sim[idx]

    # Sort the library based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the num_recom most similar library
    sim_scores = sim_scores[0:num_recommendations]

    # Get the library indices
    lib_indices = [i[0] for i in sim_scores]

    # Return the  most similar library
    # print(metadata[[ 'title','number','category','scores']].iloc[lib_indices])
    # return the number colomn as a list
    return metadata["number"].iloc[lib_indices].tolist()


def _get_similar_keywords(keywords, num_recom, metadata):
    seri = metadata["keywords"].str.split(",")
    seri = seri.apply(lambda x: [j.strip().lower() for j in x])
    scores = [
        (index, len(x) + len(keywords) - len(set(keywords + x)))
        for index, x in enumerate(seri)
    ]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    metadata["scores"] = seri.apply(
        lambda x: (len(x) + len(keywords) - len(set(keywords + x))) / len(x)
    )

    # Get the scores of the 3 most similar library
    scores = scores[0:num_recom]

    # Get the movie indices
    lib_indices = [i[0] for i in scores]

    # Return the top similar libraries
    # print(metadata[['title','number','category','scores']].iloc[lib_indices])
    return metadata[["title", "number"]].iloc[lib_indices]


def recommendation_user_based(metadata, category, num_recommendations):
    C = metadata["vote_average"].mean()
    m = metadata["votes_count"].quantile(0.1)
    q_library = metadata.copy().loc[metadata["votes_count"] >= m]

    q_library["scoreu"] = q_library.apply(_weighted_rating, args=(m, C), axis=1)

    q_library = q_library.sort_values("scoreu", ascending=False)

    # Print the top libraries
    # print(q_library[['title', 'number',  'score', 'category']].head(num_recommendations))

    df = _dataframe_related(q_library, category)
    q_df = df.sort_values("scoreu", ascending=False)
    # Print the top libraries within that category
    # print(q_df[['title', 'number',  'score', 'category']].head(num_recommendations))
    # return a list of number from q_df
    return q_df["number"].head(num_recommendations).tolist()


def recommendation_content_based(
    metadata, category, num_recommendations, question_answer
):
    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words="english")

    # Replace NaN with an empty string
    metadata["overview"] = metadata["overview"].fillna("")

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(metadata["overview"])

    # Output the shape of tfidf_matrix
    # print(tfidf_matrix.shape)

    # vocab = tfidf.vocabulary_

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # print(cosine_sim.shape)
    # Construct a reverse map of indices and library titles
    indices = pd.Series(metadata.index, index=metadata["title"]).drop_duplicates()

    df = _get_similar_keywords(
        keywords=(question_answer + category).split(),
        num_recom=num_recommendations,
        metadata=metadata,
    )
    numbers = []
    for title in df["title"]:
        numbers += _get_similar_title(
            title=title,
            num_recommendations=num_recommendations,
            cosine_sim=cosine_sim,
            indices=indices,
            metadata=metadata,
        )

    # sort the list of values in numbers based on their  occurance in the list and return the distinct values
    numbers = list(dict.fromkeys(sorted(numbers, key=numbers.count, reverse=True)))[
        :num_recommendations
    ]
    # list of library numbers
    return numbers


def recommendation_hybrid(metadata, category, num_recommendations, question_answer):
    # first create 2*num_recommendation from the content-based
    numbers = recommendation_content_based(
        metadata, category, 2 * num_recommendations, question_answer
    )
    # create a subsample of metadata which has the number category equal to numbers
    metadata = metadata[metadata["number"].isin(numbers)]
    # then create num_recommendation from the user-based (without specifying the category)
    numbers = recommendation_user_based(metadata, None, num_recommendations)
    return numbers


def recommendation_gensim(metadata, num_recommendations, question_answer):
    
    # create a corpus of all metadata overview, title, keywords
    corpus = metadata["overview"] + " " + metadata["title"] + " " + metadata["keywords"]
    # add the question answer
    corpus = pd.concat([corpus, pd.Series(question_answer)])
    # conver the pd to a unified text
    corpus = corpus.str.cat(sep=" ")

    data = preprocessing(corpus)

    # Create CBOW model
    model1 = Word2Vec(data, min_count=1, vector_size=100, window=5)
    res = []
    for i in range(len(metadata)):
        sim = similarity_sentence(
            model1,
            metadata["overview"][i]
            + " "
            + metadata["title"][i]
            + " "
            + metadata["keywords"][i],
            question_answer,
        )
        res.append((sim, metadata["number"][i]))

    # return the list of index i of the sorted res based on the sim as key
    out = sorted(res, key=lambda x: x[0], reverse=True)
    # return the second term of first k t
    numbers = [x[1] for x in out[:num_recommendations]]

    return numbers


# # Create Skip Gram model
# model2 = Word2Vec(data, min_count = 1, vector_size = 100,
#                                              window = 5, sg = 1)

# # Print results
# print("Cosine similarity between 'alice' " +
#           "and 'wonderland' - Skip Gram : ",
#     model2.wv.similarity('machine', 'learning'))


""" sub_module_dict_list is of the format:
  [{
    "id": 1,
    "title": "sample title"
    ...(same as columns in data_induced.csv)
    },
   {
    ...(next row same format)
    },
    ...(rest of modules)
  ]
"""


def main(massage, sub_module_dict_list=None, overal_recom=7):
    # create dictionary of reponces
    response = {}

    # Load metadata from csv file:
    # (UNCOMMENT NEXT 3 LINES TO USE CSV DATA)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_directory, "data_induced.csv")
    metadata = pd.read_csv(csv_file_path, low_memory=False)

    # # generate random variable for vote_average between 0,1 and votes_count between 0,100
    # metadata["vote_average"] = metadata["vote_average"].apply(
    #     lambda x: round(random.uniform(0, 1), 2)
    # )
    # metadata["votes_count"] = metadata["votes_count"].apply(
    #     lambda x:  int(random.uniform(0, 100))
    # )
    # # save the metadata to a csv file
    # metadata.to_csv(csv_file_path, index=False)

    # Load metadata from database data:
    # (COMMENT OUT NEXT LINE IF USING CSV DATA)
    # metadata = pd.DataFrame(sub_module_dict_list)

    # read the json massage to a dictionary
    massage = json.loads(massage)
    # get the number of queries from the json by couting the number of keys
    num_queries = len(massage.keys())
    for i in range(num_queries):
        # get the category of the query
        category = massage[str(i + 1)]["category"]
        approach = massage[str(i + 1)]["approach"]
        num_recommendations = massage[str(i + 1)]["num_recommendations"]

        if approach == "user_based":
            # list of library names
            lol = recommendation_user_based(metadata, category, num_recommendations)
            # print(lol)

        if approach == "content_based":
            question_answer = massage[str(i + 1)]["question_answer"]
            lol = recommendation_content_based(
                metadata, category, num_recommendations, question_answer
            )
            # print(lol)

        if approach == "hybrid":
            question_answer = massage[str(i + 1)]["question_answer"]
            lol = recommendation_hybrid(
                metadata, category, num_recommendations, question_answer
            )
            # print(lol)

        if approach == "gensim":
            question_answer = massage[str(i + 1)]["question_answer"]
            lol = recommendation_gensim(metadata, num_recommendations, question_answer)
            # print(lol)

        response[i + 1] = lol
    # create an overal response for all queries, capping at overal_recom number of recommendations
    # inlcuding the top for each querie and continue the rest based on the most common one till the overal_recom number is reached
    # get the first element of all the lists in the response dictionary
    first_picks = []
    # pick the first choice of each query
    for key in response.keys():
        first_picks.append(response[key][0])
    # create the appended of all responses
    appended_responses = []
    for req in response.keys():
        appended_responses += response[req]

    # append the first_picks again to the appended_responses to emphesize the first choices
    appended_responses += first_picks
    # get the k most common elements
    appended_responses = Counter(appended_responses).most_common(overal_recom)
    # return the list of the first element of each tuple in appended_responses
    response[0] = [x[0] for x in appended_responses]

    return response


if __name__ == "__main__":
    categories = [
        "Introduction",
        "data",
        "Supervised learning",
        "optimization",
        "Algorithm",
    ]
    approaches = ["user_based", "content_based", "hybrid", "gensim"]
    # create  a json sample massage

    massage1 = {
        "1": {
            "category": categories[2],
            "num_recommendations": 4,
            "approach": approaches[0],
        }
    }

    massage2 = {
        "1": {
            "category": categories[1],
            "num_recommendations": 4,
            "approach": approaches[1],
            "question_answer": "what is machine learning? An AI approach that ...",
        }
    }

    massage3 = {
        "1": {
            "category": categories[2],
            "num_recommendations": 4,
            "approach": approaches[3],
            "question_answer": "How to compute the learning rate? In optimization, the learning is ...",
        },
        "2": {
            "category": categories[3],
            "num_recommendations": 2,
            "approach": approaches[0],
        },
        "3": {
            "category": categories[4],
            "num_recommendations": 4,
            "approach": approaches[3],
            "question_answer": "what is machine learning? An AI approach that ...",
        },
        "4": {
            "category": categories[2],
            "num_recommendations": 4,
            "approach": approaches[3],
            "question_answer": "when applying preprocessing to the data, why we have to normalize to Guassian distribution? ...",
        },
        "5": {
            "category": categories[4],
            "num_recommendations": 4,
            "approach": approaches[3],
            "question_answer": "When we should use decision tree? It is a classification problem by using branches and divide and conquere...",
        },
        "6": {
            "category": categories[2],
            "num_recommendations": 4,
            "approach": approaches[3],
            "question_answer": "To predict labels in the classes, what methods other than SVM, and KNN can be used for training? After Spliting the data into train and test, we can use different classifiers ...",
        },
    }

    # print("approach 1")
    # massage1 = json.dumps(massage1)
    # main(massage1)
    # print("approach 2")
    # massage2 = json.dumps(massage2)
    # response = main(massage2)
    # print(response)
    print("combined")
    massage3 = json.dumps(massage3)
    response = main(massage3)
    print("returned response")
    print(response)

# Recommendation System

This folder contains the files for the recommendation system. The system includes two types of recommendation algorithms: Collaborative User-Based and Content-Based.

## Structure

- [`engine/`](engine): This folder contains the core recommendation engine.
- [`synthetic_modules.csv`](synthetic_modules.csv): This is a sample dataset used for testing the recommendation system.
- [`v0/`](v0): This folder contains the initial version of the recommendation system.

## Testing

To test the recommendation system, follow these steps:

1. Navigate to the `recommendation` directory.
2. Run the test script with the command `python test.py`.

Please ensure that you have the necessary Python environment and dependencies installed.

## Collaborative User-Based Recommendation

This type of recommendation system predicts the user's interest by collecting preferences from many users. It assumes that if two users agree on one issue, they are likely to agree on others as well.

### Input

- Context metadata = {CSV/db file, library debrief}
- Category = {the question category}
- Num recommendation n_r = {1,2,3, …}

### Output

Predict score based on user votes and return n_r top scored libraries

### Method

1. Compute C, average vote
2. Compute M, 0.1 quantile, ignore the entries one below it
3. Compute weight rating w_r using IMDB formula
4. Sort the dataframe based on w_r
5. Compute the related entries according to the category
6. Return list of size n_r from dataframe numbers with the highest score

## Content-Based Recommendation

This type of recommendation system recommends items by comparing the content of the items with a user profile. The content of each item is represented as a set of descriptors, such as the words in a document.

Please refer to the individual folders and files for more details about each component and how to use it.


### Mehtod 1: TFIDF

#### Input

- Context metadata = {CSV/db file, library debrief}
- Category = {the question category}
- Num recommendation n_r = {1,2,3, …}
- Question with the correct answer as question_answer


#### Method

1. Use TFIDF vectorizer to embed the input text
2. Remove all English stop words (the, a, …)
3. Replace Nan with empty string
4. Construct the required TF-IDF matrix by fitting and transforming the data
5. Compute the cosine similarity matrix
6. Construct a reverse map of indices and library titles
7. Create a prompt using the question_answer and category, drop duplicates
8. Obtain similar keywords
9. Split the keywords and lower case
10. Create score, tuple of index and number of matching keywords
11. Sort the scores
12. Augment the normalize score to the metadata
13. Retrieve n_r library indices and return title and number from metadata
14. Create a list of nums
15. For all n_r titles in the returned metadata, obtain similar title
16. Find index for the title compute Get the pairwise similarity scores of all library
17. Augment it to the metadata score
18. Sort the results based on score and return top n_r numbers
19. Return the list of numbers
20. Augment the numbers with the nums
21. Sort the list of values in nums based on their occurrence in the list
22. Return the distinct values

#### Output

Predict score based on similarity of question and the library overview and keywords and return n_r top scored libraries

### Mehtod 2: gensim

#### Input

- metadata (dictionary with keys "overview", "title", "keywords", "number")
- num_recommendations
- question_answer

#### Method

1. Concatenate all values of "overview", "title", and "keywords" in `metadata` to create a corpus.
2. Add question_answer to the corpus.
3. Convert the corpus to a unified text.
4. Preprocess the unified text (e.g., tokenize, lowercase).
5. Create a Word2Vec model (`model1`) with `data` from step 5.
6. Initialize an empty list `res` for storing similarity scores and numbers.
7. For each item in `metadata`:
   - Calculate the similarity between the item's "overview", "title", "keywords" (preprocessed) and `question_answer` (preprocessed) using `model1`.
   - Append the similarity score and item number to `res`.
8. Sort `res` in descending order based on similarity scores.
9. Extract the numbers of the first `num_recommendations` items from `res`.

#### Output

- List of numbers of the recommended items.
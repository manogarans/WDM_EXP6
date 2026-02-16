### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 16-02-2026
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

```python
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from tabulate import tabulate

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Added to resolve the LookupError

# -------------------------------
# Document Collection
# -------------------------------
documents = {
    "doc1": "the health Observation for March",
    "doc2": "the health oriented Calender",
    "doc3": "the awareness news for March awareness",
}

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        token for token in tokens
        if token not in stopwords.words("english")
        and token not in string.punctuation
    ]
    return " ".join(tokens)

preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}

# -------------------------------
# Vectorization
# -------------------------------
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(preprocessed_docs.values())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

terms = tfidf_vectorizer.get_feature_names_out()

# -------------------------------
# Display TF Table
# -------------------------------
print("\n--- Term Frequencies (TF) ---\n")
tf_table = count_matrix.toarray()

print(tabulate(
    [["Doc ID"] + list(terms)] +
    [[list(preprocessed_docs.keys())[i]] + list(row)
     for i, row in enumerate(tf_table)],
    headers="firstrow",
    tablefmt="grid"
))

# -------------------------------
# DF & IDF
# -------------------------------
df = np.sum(count_matrix.toarray() > 0, axis=0)
idf = tfidf_vectorizer.idf_

df_idf_table = []
for i, term in enumerate(terms):
    df_idf_table.append([term, df[i], round(idf[i], 4)])

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")
print(tabulate(
    df_idf_table,
    headers=["Term", "Document Frequency (DF)", "Inverse Document Frequency (IDF)"],
    tablefmt="grid"
))

# -------------------------------
# TF-IDF Table
# -------------------------------
print("\n--- TF-IDF Weights ---\n")
tfidf_table = tfidf_matrix.toarray()

print(tabulate(
    [["Doc ID"] + list(terms)] +
    [[list(preprocessed_docs.keys())[i]] +
     list(map(lambda x: round(x, 4), row))
     for i, row in enumerate(tfidf_table)],
    headers="firstrow",
    tablefmt="grid"
))

# -------------------------------
# Cosine Similarity + Search
# -------------------------------
def cosine_similarity_search(query, tfidf_matrix, tfidf_vectorizer, documents, preprocessed_docs):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query]).toarray()[0]

    results = []

    for idx, doc_vector in enumerate(tfidf_matrix.toarray()):
        doc_id = list(preprocessed_docs.keys())[idx]
        doc_text = documents[doc_id]

        dot_product = np.dot(query_vector, doc_vector)
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)

        similarity = dot_product / (norm_q * norm_d) if norm_q != 0 and norm_d != 0 else 0.0

        results.append([
            doc_id,
            doc_text,
            round(dot_product, 4),
            round(norm_q, 4),
            round(norm_d, 4),
            round(similarity, 4)
        ])

    results.sort(key=lambda x: x[5], reverse=True)
    return results, query_vector

# -------------------------------
# User Query
# -------------------------------
query = input("\nEnter your query: ")

results_table, query_vector = cosine_similarity_search(
    query,
    tfidf_matrix,
    tfidf_vectorizer,
    documents,
    preprocessed_docs
)

# -------------------------------
# Display Results
# -------------------------------
print("\n--- Search Results and Cosine Similarity ---\n")

headers = [
    "Doc ID",
    "Document",
    "Dot Product",
    "Query Magnitude",
    "Doc Magnitude",
    "Cosine Similarity"
]

print(tabulate(results_table, headers=headers, tablefmt="grid"))

# -------------------------------
# Query TF-IDF Weights
# -------------------------------
print("\n--- Query TF-IDF Weights ---\n")

query_weights = [
    (terms[i], round(query_vector[i], 4))
    for i in range(len(terms))
    if query_vector[i] > 0
]

print(tabulate(
    query_weights,
    headers=["Term", "Query TF-IDF Weight"],
    tablefmt="grid"
))

# -------------------------------
# Ranked Documents
# -------------------------------
print("\n--- Ranked Documents ---\n")

ranked_docs = []
for idx, res in enumerate(results_table, start=1):
    ranked_docs.append([
        idx,
        res[0],
        res[1],
        res[5]
    ])

print(tabulate(
    ranked_docs,
    headers=["Rank", "Document ID", "Document Text", "Cosine Similarity"],
    tablefmt="grid"
))

highest_doc = max(results_table, key=lambda x: x[5])
print(f"\nThe highest rank cosine score is: {highest_doc[5]} (Document ID: {highest_doc[0]})")

```
### Output :
<img width="904" height="603" alt="Screenshot 2026-02-16 105103" src="https://github.com/user-attachments/assets/b3cd21dc-01a7-49ae-aa3f-0452c889717e" />
<img width="1028" height="677" alt="Screenshot 2026-02-16 105115" src="https://github.com/user-attachments/assets/4c406860-e1ee-40b0-82f0-688f6ec5dfd5" />


### Result:

Thus the, Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, TF-IDF scores, and performing similarity calculations between queries and documents is executed successfully.

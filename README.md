### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 14.02.2026
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
~~~
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from tabulate import tabulate

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Added to download the missing resource

# Sample documents
documents = {
    "doc1": "The cat sat on the mat",
    "doc2": "The dog sat on the log",
    "doc3": "The cat lay on the rug",
    
}

# Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    return " ".join(tokens)

preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# Vectorizers
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(preprocessed_docs.values())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

terms = tfidf_vectorizer.get_feature_names_out()

# Term Frequency Table
print("\n--- Term Frequencies (TF) ---\n")
tf_table = count_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(row) for i, row in enumerate(tf_table)], headers="firstrow", tablefmt="grid"))

# Document Frequency (DF) and IDF Table
df = np.sum(count_matrix.toarray() > 0, axis=0)
idf = tfidf_vectorizer.idf_

df_idf_table = []
for i, term in enumerate(terms):
    df_idf_table.append([term, df[i], round(idf[i], 4)])

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")
print(tabulate(df_idf_table, headers=["Term", "Document Frequency (DF)", "Inverse Document Frequency (IDF)"], tablefmt="grid"))

# TF-IDF Table
print("\n--- TF-IDF Weights ---\n")
tfidf_table = tfidf_matrix.toarray()
print(tabulate([["Doc ID"] + list(terms)] + [[list(preprocessed_docs.keys())[i]] + list(map(lambda x: round(x, 4), row)) for i, row in enumerate(tfidf_table)], headers="firstrow", tablefmt="grid"))

# Manual Cosine Similarity calculation
def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0.0
    return dot_product, norm_vec1, norm_vec2, similarity

# Search function
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query]).toarray()[0]
    results = []

    for idx, doc_vector in enumerate(tfidf_matrix.toarray()):
        doc_id = list(preprocessed_docs.keys())[idx]
        doc_text = documents[doc_id]
        dot, norm_q, norm_d, sim = cosine_similarity_manual(query_vector, doc_vector)
        results.append([doc_id, doc_text, round(dot, 4), round(norm_q, 4), round(norm_d, 4), round(sim, 4)])

    results.sort(key=lambda x: x[5], reverse=True)
    return results, query_vector

# Input from user
query = input("\nEnter your query: ")

# Perform search
results_table, query_vector = search(query, tfidf_matrix, tfidf_vectorizer)

# Display Cosine Similarity Table
print("\n--- Search Results and Cosine Similarity ---\n")
headers = ["Doc ID", "Document", "Dot Product", "Query Magnitude", "Doc Magnitude", "Cosine Similarity"]
print(tabulate(results_table, headers=headers, tablefmt="grid"))
# Display Query TF-IDF Weights
print("\n--- Query TF-IDF Weights ---\n")
query_weights = [(terms[i], round(query_vector[i], 4)) for i in range(len(terms)) if query_vector[i] > 0]
print(tabulate(query_weights, headers=["Term", "Query TF-IDF Weight"], tablefmt="grid"))

# Display Ranking
print("\n--- Ranked Documents ---\n")
ranked_docs = []
for idx, res in enumerate(results_table, start=1):
    ranked_docs.append([idx, res[0], res[1], res[5]])

print(tabulate(ranked_docs, headers=["Rank", "Document ID", "Document Text", "Cosine Similarity"], tablefmt="grid"))
# Find the document with the highest cosine similarity
highest_doc = max(results_table, key=lambda x: x[5])  # x[5] is the cosine similarity
highest_doc_id = highest_doc[0]
highest_doc_text = highest_doc[1]
highest_score = highest_doc[5]

print(f"\nThe highest rank cosine score is: {highest_score} (Document ID: {highest_doc_id})")
~~~
### Output:
<img width="1907" height="1120" alt="Screenshot 2026-02-13 160455" src="https://github.com/user-attachments/assets/943fada7-eb8e-44fa-aa9a-28af68ed5a6a" />
<img width="1848" height="986" alt="Screenshot 2026-02-14 140511" src="https://github.com/user-attachments/assets/414b6162-5557-46d6-8bce-02570d4f47e3" />
<img width="1890" height="1023" alt="Screenshot 2026-02-14 140535" src="https://github.com/user-attachments/assets/9d2911e4-e72f-4a62-96cb-faa05e9ff3e1" />
<img width="1903" height="1004" alt="Screenshot 2026-02-14 140610" src="https://github.com/user-attachments/assets/8ad38ef6-1a8e-4dd3-bfb3-0742acbb7f5f" />

### Result:
Thus, the implementation of Information Retrieval Using Vector Space Model in Python is executed successfully.

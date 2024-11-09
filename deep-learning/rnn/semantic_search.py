
# In this notebook, we use sentence embeddings to perform semantic search.
# We utilize the Universal Sentence Encoder (USE) model to encode both questions and contexts/answers (paragraphs) as embeddings.
# USE is a pre-trained model that generates semantic embeddings for small to medium-sized text inputs.
# For better embeddings but slower performance, transformer-based models like BERT can be used.
# Then, using a question/answer dataset, we perform semantic search to retrieve the most relevant contexts based on a query.
# We search for similarity between the encodings of the query and both the contexts and questions, and return the most relevant ones.

# We use the SQuAD (Stanford Question Answering Dataset) [https://rajpurkar.github.io/SQuAD-explorer/]
# SQuAD is one of the most widely used datasets for question-answering tasks.
# It consists of questions posed by crowdworkers on Wikipedia articles, with the corresponding answers as text
# spans within the articles.

#!pip install tensorflow tensorflow-hub datasets


from datasets import load_dataset
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the SQuAD dataset (using only the 'train' split for demonstration)
squad_dataset = load_dataset("squad", split="train")

# Extract contexts (paragraphs) and corresponding questions
contexts = squad_dataset["context"]
questions = squad_dataset["question"]

# Display the first few entries for reference
print(f"First context: {contexts[0]}")
print(f"First question: {questions[0]}")

# Load the USE model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# For the semantic search, we'll encode questions dynamically on each query
def semantic_search(query_p: str, contexts_p: np.array, questions_p: np.array,
                    top_k: int, questions_embeddings_p: np.array, context_embeddings_p: np.array) \
        -> list[dict[str, str]]:
    """
    This function performs semantic search to retrieve the most relevant contexts based on a query.
    If questions_embeddings_p is provided, it searches by questions.
    Otherwise, it searches by contexts.
    :param query_p: the query to be semantically searched
    :param contexts_p: the contexts (paragraphs) to search from
    :param questions_p: the questions to search from
    :param top_k: how many results to return
    :param questions_embeddings_p: the embeddings of the questions (if searching by questions; otherwise None)
    :param context_embeddings_p: the embeddings of the contexts (if searching by contexts; otherwise None)
    :return: a list of dictionaries containing the most relevant contexts ('context'),
    sample questions ('sample_question'), and similarity scores ('similarity')
    """
    # Encode the query
    query_embedding = embed([query_p])
    # this function allow searching by question or by context
    semantic_embeddings = questions_embeddings_p if questions_embeddings_p is not None else context_embeddings_p
    # Compute cosine similarities between the query and all queries embeddings
    similarities = cosine_similarity(query_embedding, semantic_embeddings).flatten()
    # Get the indices of the top_k most similar contexts
    # argsort returns the indices that would sort an array ascending, [-top_k:] gets the top_k largest values,
    # and [::-1] reverses the order
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    # Retrieve the most similar contexts and their corresponding questions
    results = []
    for idx in top_k_indices:
        results.append({
            'context': contexts_p[idx],
            'sample_question': questions_p[idx],  # A sample question related to this context
            'similarity': similarities[idx]
        })
    return results


# Example queries
queries = ['What is the capital of the United States of America?',
           'What language is spoken in Andorra?',
           'Who was Martin Luther King?']

# Encode the contexts (paragraphs) and questions. This may take some time.
questions_embeddings = embed(questions)
contexts_embeddings = embed(contexts)


def show_results(search_results_p: list[dict[str, str]]) -> None:
    """
    Display the search results
    :param search_results_p: the search results to display
    """
    for i, result in enumerate(search_results_p):
        print(f"\tResult {i + 1}")
        print(f"\tContext: {result['context']}")
        print(f"\tSample Question: {result['sample_question']}")
        print(f"\tSimilarity Score: {result['similarity']:.4f}", "\n")


for query in queries:
    print(f"Query: {query}")
    print("Question-based search:")
    # Get top 3 most relevant
    search_results = semantic_search(query, contexts, questions, 3, questions_embeddings, None)
    show_results(search_results)
    print("Question-based search:")
    # Get top 3 most relevant
    search_results = semantic_search(query, contexts, questions, 3, None, contexts_embeddings)
    # Display the results
    show_results(search_results)
    print("-" * 50)

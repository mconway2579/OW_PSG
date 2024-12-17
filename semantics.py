from openai import OpenAI
import numpy as np
from config import embedding_model, node_match_thresh, edge_match_thresh
from APIKeys import API_KEY
from sentence_transformers import SentenceTransformer, util


client = OpenAI(
        api_key= API_KEY,
    )
Sentence_Encoder = SentenceTransformer('all-MiniLM-L6-v2')
print(f"{Sentence_Encoder.device=}")



def get_openai_embeddings(str1, str2):
    # Get embeddings for the strings
    response1 = client.embeddings.create(input=str1, model=embedding_model)
    embedding1 = response1.data[0].embedding  # Access the embedding attribute

    response2 = client.embeddings.create(input=str2, model=embedding_model)
    embedding2 = response2.data[0].embedding  # Access the embedding attribute
    # Calculate cosine similarity
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    return emb1, emb2

def get_transformer_embedding(str1, str2):
    embeddings = Sentence_Encoder.encode([str1, str2])
    return embeddings


def str_semantic_distance(str1, str2, encoder_func = get_transformer_embedding):
    assert isinstance(str1, str), f"The variable {str1=} is not a string!"
    assert isinstance(str2, str), f"The variable {str2=} is not a string!"

    emb1, emb2 = encoder_func(str1, str2)
    cosin_sim = min(util.cos_sim(emb1, emb2).item(), 1)

    distance = 1 - cosin_sim
    assert distance >= 0, f"{str1}, {str2}: {cosin_sim=} {distance=}"
    return distance


def node_replacement_cost(N1, N2):
    N1 = N1.get("name")
    N2 = N2.get("name")
    dist = str_semantic_distance(N1, N2)
    return dist

def node_match_func(N1, N2,):
    dist = node_replacement_cost(N1, N2)
    return dist <= node_match_thresh

def edge_replacement_cost(E1, E2):
    E1 = E1.get("name")
    E2 = E2.get("name")
    dist = str_semantic_distance(E1, E2)
    return dist

def edge_match_func(E1, E2):
    dist = edge_replacement_cost(E1, E2)
    return dist <= edge_match_thresh



if __name__ == "__main__":
    import itertools

    strings = ["red block", "red cube", "blue block", "pink block", "pink cube", "paper", "white paper"]
    

    for pair in itertools.combinations_with_replacement(strings, 2):
        trans_distance = str_semantic_distance(pair[0], pair[1], encoder_func=get_transformer_embedding)
        oai_distance = str_semantic_distance(pair[0], pair[1], encoder_func=get_openai_embeddings)

        print(f"{pair[0]}, {pair[1]}: {trans_distance}, {oai_distance}")
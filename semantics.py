from openai import OpenAI
import numpy as np
from config import gpt_embedding_model, node_match_thresh, edge_match_thresh, st_embedding_model
from APIKeys import API_KEY
from sentence_transformers import SentenceTransformer, util
import networkx as nx

client = OpenAI(
        api_key= API_KEY,
    )
Sentence_Encoder = SentenceTransformer(st_embedding_model)
print(f"{Sentence_Encoder.device=}")



def get_openai_embeddings(str1, str2):
    # Get embeddings for the strings
    response1 = client.embeddings.create(input=str1, model=gpt_embedding_model)
    embedding1 = response1.data[0].embedding  # Access the embedding attribute

    response2 = client.embeddings.create(input=str2, model=gpt_embedding_model)
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

def find_semantic_matching(str_list_1, str_list_2, thresh):
    print(f"\nbegin matching")
    bi_partite_graph = nx.Graph()
    for str1 in str_list_1:
        bi_partite_graph.add_node(f"g1_{str1}", name=str1)
    for str2 in str_list_2:
        bi_partite_graph.add_node(f"g2_{str2}", name=str2)

    for str1 in str_list_1:
        for str2 in str_list_2:
            node_semantic_distance = str_semantic_distance(str1, str2)
            bi_partite_graph.add_edge(f"g1_{str1}", f"g2_{str2}", weight = node_semantic_distance)

    matching = nx.algorithms.matching.min_weight_matching(bi_partite_graph, weight="weight")
    a2b = {}
    b2a = {}
    for match in matching:
        edge_data = bi_partite_graph.get_edge_data(match[0], match[1])
        g1_match = match[0] if match[0].startswith("g1_") else match[1]
        g2_match = match[0] if match[0].startswith("g2_") else match[1]
        
        if edge_data['weight'] < thresh:
            a2b[g1_match[3:]] = g2_match[3:]
            b2a[g2_match[3:]] = g1_match[3:]
            print(f"Kept ({g1_match[3:]} matches to {g2_match[3:]}) with score {edge_data['weight']}")
        else:
            print(f"Rejected ({g1_match[3:]} matches to {g2_match[3:]}) with score {edge_data['weight']}")


    print()
    return a2b, b2a

if __name__ == "__main__":
    import itertools

    strings_A = ["red block", "red cube", "blue block", "pink block", "pink cube", "paper", "white paper"]

    

    #for pair in itertools.combinations_with_replacement(strings_A, 2):
    #    trans_distance = str_semantic_distance(pair[0], pair[1], encoder_func=get_transformer_embedding)
    #    oai_distance = str_semantic_distance(pair[0], pair[1], encoder_func=get_openai_embeddings)
    #
    #    print(f"{pair[0]}, {pair[1]}: {trans_distance}, {oai_distance}")

    strings_B = ["crimson block", "scarlet cube", "navy block", "magenta block", "rose cube", "notebook paper", "blank paper"]
    a2b, b2a = find_semantic_matching(strings_A, strings_B, node_match_thresh)
    print("\na2b")
    for key, value in a2b.items():
        print(f"{key}:{value}")

    print("\nb2a")
    for key, value in b2a.items():
        print(f"{key}:{value}")
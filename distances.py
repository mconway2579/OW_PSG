import numpy as np
from config import node_match_thresh, str_match_thresh, st_embedding_model, vit_str, str_dist_weight
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from torch import nn
import torch
class Sentence_Encoder(nn.Module):
    def __init__(self):
        super(Sentence_Encoder, self).__init__()

        self.model = SentenceTransformer(st_embedding_model)
        self.model.to(torch.device("cuda")) if torch.cuda.is_available() else None
        self.model.to(torch.device("mps")) if torch.backends.mps.is_available() else None
        self.device = self.model.device

    def encode(self, strings):
        embeddings = self.model.encode(strings)
        return embeddings

    def get_device(self):
        return self.device
SE = Sentence_Encoder()
print(f"{SE.model.device=}")

class VitEncoder(nn.Module):
    def __init__(self, model_str=vit_str):
        super(VitEncoder, self).__init__()
        self.image_processor = ViTImageProcessor.from_pretrained(model_str)
        self.model = ViTModel.from_pretrained(model_str, add_pooling_layer=False)
        self.model.to(torch.device("cuda")) if torch.cuda.is_available() else None
        self.model.to(torch.device("mps")) if torch.backends.mps.is_available() else None
    def encode(self, images):
        converted_imgs = []
        for i, np_img in enumerate(images):
            if np_img.max() <= 1.0:
                np_img = (np_img * 255).astype('uint8')
            else:
                np_img = np_img.astype('uint8')
            converted_imgs.append(Image.fromarray(np_img).resize((224, 224)))

        inputs = self.image_processor(images=converted_imgs, return_tensors="pt")
        device = self.get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token
    def get_device(self):
        return next(self.parameters()).device
VE = VitEncoder()
print(f"{VE.model.device=}")

def cos_dist(emb1, emb2):
    cosin_sim = min(util.cos_sim(emb1, emb2).item(), 1) #when is this ever over 1?
    distance = 1 - cosin_sim
    return distance
def str_dist(str1, str2):
    emb1, emb2 = SE.encode([str1, str2])
    str_dist = cos_dist(emb1, emb2)
    return str_dist
def str_node_distance(N1_data, N2_data):
    Name1 = N1_data.str_label
    Name2 = N2_data.str_label
    return str_dist(Name1, Name2)
def vis_node_distance(N1_data, N2_data):
    average_encoding_N1 = VE.encode(N1_data.images).mean(dim=0)
    average_encoding_N2 = VE.encode(N2_data.images).mean(dim=0)
    #print(f"{average_encoding_N1.shape=}, {average_encoding_N2.shape=}")
    vis_dist = cos_dist(average_encoding_N1, average_encoding_N2)
    return vis_dist
def data_node_distance(N1_data, N2_data):
    str_dist = str_node_distance(N1_data, N2_data)
    vis_dist = vis_node_distance(N1_data, N2_data)
    return (vis_dist*(1-str_dist_weight)) + (str_dist * str_dist_weight)


def edge_d(E1, E2):
    Edge1 = E1.get("name")
    Edge2 = E2.get("name")
    emb1, emb2 = SE.encode([Edge1, Edge2])
    str_dist = cos_dist(emb1, emb2)
    return str_dist

def find_datanode_matching(node_list_1, node_list_2):
    str_list_1 = [n[1]["data"].str_label for n in node_list_1]
    str_list_2 = [n[1]["data"].str_label for n in node_list_2]
    print(f"\nbegin data matching between\n   {str_list_1}\n   {str_list_2}")
    bi_partite_graph = nx.Graph()
    for n1 in node_list_1:
        #print(f"\n\n{n1=} \n{dir(n1)=}")
        bi_partite_graph.add_node(f"g1_{n1[1]["data"].str_label}", data=n1[1]["data"])
    for n2 in node_list_2:
        bi_partite_graph.add_node(f"g2_{n2[1]["data"].str_label}", data=n2[1]["data"])



    for n1 in node_list_1:
        for n2 in node_list_2:
            #print(f"Comparing {n1[1]['data'].str_label} to {n2[1]['data'].str_label}")
            distance = data_node_distance(n1[1]['data'], n2[1]['data'])
            bi_partite_graph.add_edge(f"g1_{n1[1]['data'].str_label}", f"g2_{n2[1]['data'].str_label}", weight = distance)

    matching = nx.algorithms.matching.min_weight_matching(bi_partite_graph, weight="weight")
    a2b = {}
    b2a = {}
    for match in matching:
        edge_data = bi_partite_graph.get_edge_data(match[0], match[1])
        g1_match = match[0] if match[0].startswith("g1_") else match[1]
        g2_match = match[0] if match[0].startswith("g2_") else match[1]
        
        if edge_data['weight'] < node_match_thresh:
            a2b[g1_match[3:]] = g2_match[3:]
            b2a[g2_match[3:]] = g1_match[3:]
            print(f"Kept ({g1_match[3:]} matches to {g2_match[3:]}) with score {edge_data['weight']}")
        else:
            print(f"Rejected ({g1_match[3:]} matches to {g2_match[3:]}) with score {edge_data['weight']}")


    print()
    return a2b, b2a
def find_str_matching(list_A, list_B):
    print(f"\nbegin matching strings")
    bi_partite_graph = nx.Graph()
    for str_a in list_A:
        #print(f"\n\n{n1=} \n{dir(n1)=}")
        bi_partite_graph.add_node(f"g1_{str_a}")

    for str_b in list_B:
        bi_partite_graph.add_node(f"g2_{str_b}")

    for str_a in list_A:
        for str_b in list_B:
            #print(f"Comparing {n1} to {n2}")
            distance = str_dist(str_a, str_b)
            bi_partite_graph.add_edge(f"g1_{str_a}", f"g2_{str_b}", weight = distance)

    matching = nx.algorithms.matching.min_weight_matching(bi_partite_graph, weight="weight")
    a2b = {}
    b2a = {}
    for match in matching:
        edge_data = bi_partite_graph.get_edge_data(match[0], match[1])
        g1_match = match[0] if match[0].startswith("g1_") else match[1]
        g2_match = match[0] if match[0].startswith("g2_") else match[1]
        
        if edge_data['weight'] < str_match_thresh:
            a2b[g1_match[3:]] = g2_match[3:]
            b2a[g2_match[3:]] = g1_match[3:]
            print(f"Kept ({g1_match[3:]} matches to {g2_match[3:]}) with score {edge_data['weight']}")
        else:
            print(f"Rejected ({g1_match[3:]} matches to {g2_match[3:]}) with score {edge_data['weight']}")


    print()
    return a2b, b2a
if __name__ == "__main__":
    import itertools
    import pickle

    from APIKeys import API_KEY
    from openai import OpenAI
    from SceneGraphGeneration import *

    sam = SAM2()
    print(f"{sam=}")
    owl = OWLv2()
    print(f"{owl=}")
    
    client = OpenAI(
        api_key= API_KEY,
    )

    with open("./custom_dataset/one on two/top_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
    K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])

    
    prompt = "how are objects layed out on the table?"
    graph1 = get_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale, prompt)
    nodes1 = list(graph1.nodes(data=True))

    with open("./custom_dataset/one on two/angled_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
    K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])

    prompt = "how are objects layed out on the table?"
    graph2 = get_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale, prompt)
    nodes2 = list(graph2.nodes(data=True))

    sentence_encodings = SE.encode(["hello", "world"])
    print(f"{sentence_encodings.shape=}")
    image_ecodings = VE.encode([rgb_img, rgb_img])
    print(f"{image_ecodings.shape=}")

    test_str_dist = str_dist("hello", "world")
    print(f"{test_str_dist=}")

    a2b, b2a = find_datanode_matching(nodes1, nodes2)
    print("\na2b")
    for key, value in a2b.items():
        print(f"{key}:{value}")

    print("\nb2a")
    for key, value in b2a.items():
        print(f"{key}:{value}")


    list_A = [
        "Happy",
        "Sad",
        "Angry",
        "Excited",
        "Bored",
        "Tired",
        "Anxious",
        "Calm",
        "Surprised",
        "Confident"
    ]
    list_B = [
        "Joyful",
        "Melancholy",
        "Irate",
        "Enthusiastic",
        "Disinterested",
        "Weary",
        "Nervous",
        "Peaceful",
        "Astonished",
        "Self-assured"
    ]
    a2b, b2a = find_str_matching(list_A, list_B)
    print("\na2b")
    for key, value in a2b.items():
        print(f"{key}:{value}")

    print("\nb2a")
    for key, value in b2a.items():
        print(f"{key}:{value}")
from SceneGraphGeneration import get_graph, Node, semantic_graph_from_json, point_clound_graph_from_json
from helper_functions import get_points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import networkx as nx
import pickle
from config import N_graph_samples
from gpt_states import get_state


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def bert_similarity(str1, str2):
    embeddings = model.encode([str1, str2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def node_match_func(N1, N2, node_sim_thresh=0.8):
    try:
        N1 = N1.get("name")
        N2 = N2.get("name")
    except:
        print(f"Try except triggered trying to get names {N1=}, {N2=}")
    #print(f"{N1=}, {N2=}")
    embeddings = model.encode([N1, N2])
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    #if N1 != N2 and score > 0.5:
    #    print(f"{N1=}, {N2=}, {score=}")
    return score > node_sim_thresh


def edge_match_func(E1, E2, edge_sim_thresh=0.8):
    E1 = E1.get("connection")
    E2 = E2.get("connection")

    embeddings = model.encode([E1, E2])
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    #if E1 != E2 and score > 0.5:
    #    print(f"{E1=}, {E2=}, {score=}")
    return score > edge_sim_thresh


class Graph_Manager:
    def __init__(self):
        self.graph_history = []
        self.last_graphed = None

        self.fig = plt.figure(figsize = (19,12))
        self.ax2d_rgb = self.fig.add_subplot(221)
        self.ax2d_depth = self.fig.add_subplot(222)
        self.ax2d_graph = self.fig.add_subplot(223)
        self.ax3d = self.fig.add_subplot(224, projection='3d')
        self.last_rgb = None
        self.last_depth = None
        self.start()

    def update_display(self, rubbish):
        if len(self.graph_history) == 0:
            return
        if self.last_graphed is not None and len(self.graph_history) == self.last_graphed:
            return
        G = self.graph_history[-1]
        """
        G = nx.DiGraph()#self.graph_history[-1]
        for g in self.graph_history:
            for node, attrs in g.nodes(data=True):
                #print(f"{node=}, {attrs=}")
                if node in G:
                    G.nodes[node]["data"] = G.nodes[node]["data"] + attrs["data"]
                else:
                    G.add_node(node, data=attrs['data'])

            for u, v, attrs in g.edges(data=True):
                #print(f"{u=}, {v=}, {attrs=}")
                if (u,v) in G.edges:
                    G.edges[u,v]["connection"] = G.edges[u,v]["connection"] + ", "+ attrs["connection"]
                else:
                    G.add_edge(u, v, connection=attrs['connection'])
            G.graph["timestamp"] = g.graph["timestamp"]
            G.graph["rgb_img"] = g.graph["rgb_img"]
            G.graph["depth_img"] = g.graph["depth_img"]
        """
        points, colors = get_points(G)

        self.ax3d.clear()
        self.ax3d.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=0.5)

        self.ax2d_rgb.clear()
        self.ax2d_rgb.imshow(G.graph['rgb_img'])
        
        self.ax2d_depth.clear()
        self.ax2d_depth.imshow(G.graph['depth_img'])

        self.ax2d_graph.clear()

        pos = nx.spring_layout(G)
        # Draw the graph on the specified axes
        nx.draw(G, pos=pos, ax=self.ax2d_graph, with_labels=True, node_color='lightblue', node_size=1500, font_size=15)
        # Draw edge labels on the same axes
        edge_labels = nx.get_edge_attributes(G, 'connection')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=12, ax=self.ax2d_graph)
        # Show without blocking, then pause
        plt.tight_layout()

        self.last_graphed = len(self.graph_history)


        #plt.draw()

    def add_graph(self, client, owl, sam, rgb_img, depth_img, pose, K, depth_scale, user_prompt):
        states = []
        for i in range(N_graph_samples):
            _, state_json, _, _ = get_state(client, rgb_img, user_prompt)
            print(state_json)
            print()
            states.append(state_json)
        graphs = []
        for state in states:
            graphs.append(semantic_graph_from_json(state))
        
        distance_matrix = np.zeros((len(graphs), len(graphs)))
        acc = 0
        for i, g1 in enumerate(graphs):
            for j, g2 in enumerate(graphs):
                #N = input("enter key to match on: ")
                #node_match_func(g1.nodes[N], g2.nodes[N])
                ged = nx.graph_edit_distance(g1, g2, node_match=node_match_func, edge_match=edge_match_func)
                #ged = np.inf
                #for v in nx.optimize_graph_edit_distance(g1, g2):
                #    if v < ged:
                #        ged = v
                distance_matrix[i,j] = ged
                acc +=1
                print(f"forming distance matrix {(i*len(graphs)) + j+1}/{len(graphs)**2}", end="\r")

        print(f"distance_matrix=\n{distance_matrix}")
        distance_sums = np.sum(distance_matrix, axis=1)
        print(f"distance_sums=\n{distance_sums}")
        closest_idx = np.argmin(distance_sums)
        closest_state = states[closest_idx]

        print("forming pointcloud graph")
        G = point_clound_graph_from_json(closest_state, rgb_img, depth_img, pose, owl, sam, K, depth_scale)
        
        self.graph_history.append(G)
        return G

    def start(self):
        self.animation = FuncAnimation(self.fig, self.update_display, interval=1000, cache_frame_data=False)  # 1 Hz update
        plt.show(block = False)
        plt.pause(1)
if __name__ == "__main__":
    from APIKeys import API_KEY
    import os
    from SceneGraphGeneration import intrinsic_obj, OWLv2, SAM2
    from openai import OpenAI

    top_dir = f"./custom_dataset/"
    def find_pkl_files(top_directory):
        pkl_files = []
        for root, _, files in os.walk(top_directory):
            for file in files:
                if file.endswith(".pkl"):
                    # Get the absolute path
                    absolute_path = os.path.abspath(os.path.join(root, file))
                    pkl_files.append(absolute_path)
        return pkl_files
    samples = find_pkl_files(top_dir)

    owl = OWLv2()
    print(f"{owl=}")
    sam = SAM2()
    print(f"{sam=}")

    client = OpenAI(
        api_key= API_KEY,
    )

    gm = Graph_Manager()
    inp = "a"
    i = 0
    while inp != "q":
        sample = samples[i]
        with open(sample, "rb") as file:
            rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
            K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])
        
        gm.add_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale, "how are objects layed out on the table? I know some of the objects are blocks")
        inp = input("press q to quit: ")
        i+=1

    
    
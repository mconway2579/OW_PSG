from SceneGraphGeneration import get_points, get_graph, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import networkx as nx
import pickle
from config import N_graph_samples
from gpt_states import get_state
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

    def add_graph(self, client, owl, sam, rgb_img, depth_img, pose, K, depth_scale):
        print("begin add graph")
        states = []
        for i in range(N_graph_samples):
            _, state_json, _, _ = get_state(client, rgb_img)
            print(state_json)
            print()
            states.append(state_json)
        graphs = []
        for state in states:
            G = nx.DiGraph()
            for obj_str in state['objects']:
                G.add_node(obj_str)
            for relation in state['object_relationships']:
                G.add_edge(relation[0], relation[2], connection=relation[1])
            graphs.append(G)
        
        distance_matrix = np.zeros((len(graphs), len(graphs)))
        for i, g1 in enumerate(graphs):
            for j, g2 in enumerate(graphs):
                ged = nx.graph_edit_distance(g1, g2)
                distance_matrix[i,j] = ged
        print(f"distance_matrix=\n{distance_matrix}")
        distance_sums = np.sum(distance_matrix, axis=1)
        print(f"distance_sums=\n{distance_sums}")
        closest_idx = np.argmin(distance_sums)
        closest_state = states[closest_idx]

        G = nx.DiGraph()
        G.graph["timestamp"] = time.time()
        G.graph["observation_pose"] = pose
        G.graph["rgb_img"] = rgb_img
        G.graph["depth_img"] = depth_img

        for object in closest_state["objects"]:
            obj_node = Node(object, rgb_img, depth_img, owl, sam, K, depth_scale, pose)
            G.add_node(object, data=obj_node)

        for edge in closest_state["object_relationships"]:
            G.add_edge(edge[0], edge[2], connection=edge[1])
        
        self.graph_history.append(G)
        return G




            


    def start(self):
        self.animation = FuncAnimation(self.fig, self.update_display, interval=1000, cache_frame_data=False)  # 1 Hz update
        plt.show(block = False)
        plt.pause(1)
if __name__ == "__main__":
    from APIKeys import API_KEY
    import os
    import cv2
    import random
    from SceneGraphGeneration import intrinsic_obj, OWLv2, SAM2
    from openai import OpenAI
    from magpie_control.ur5 import homog_coord_to_pose_vector


    
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

        gm.add_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale)
        inp = input("press q to quit: ")
        i+=1

    
    
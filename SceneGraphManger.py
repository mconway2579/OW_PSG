from SceneGraphGeneration import get_points, get_graph, Node
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import networkx as nx
import pickle

class Graph_Manager:
    def __init__(self):
        self.graph_history = []
        self.last_graphed = None

        self.fig = plt.figure(figsize = (12,12))
        self.ax2d_rgb = self.fig.add_subplot(221)
        self.ax2d_depth = self.fig.add_subplot(222)
        self.ax2d_graph = self.fig.add_subplot(223)
        self.ax3d = self.fig.add_subplot(224, projection='3d')
        self.start()


    def update_display(self, rubbish):
        if len(self.graph_history) == 0:
            return
        if self.last_graphed is not None and self.graph_history[-1] == self.last_graphed:
            return
        
        G = self.graph_history[-1]

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

        self.last_graphed = G


        #plt.draw()

    def add_graph(self, graph):
        self.graph_history.append(graph)

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

        graph = get_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale)

        gm.add_graph(graph)
        inp = input("press q to quit: ")
        i+=1

    
    
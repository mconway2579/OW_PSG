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
        self.ax2d = self.fig.add_subplot(211)
        self.ax3d = self.fig.add_subplot(212, projection='3d')
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

        self.ax2d.clear()

        pos = nx.spring_layout(G)
        # Draw the graph on the specified axes
        nx.draw(G, pos=pos, ax=self.ax2d, with_labels=True, node_color='lightblue', node_size=1500, font_size=15)
        # Draw edge labels on the same axes
        edge_labels = nx.get_edge_attributes(G, 'connection')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=12, ax=self.ax2d)
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


    
    top_dir = f"/home/max/OW_PSG/SUNRGBD/kv1/b3dodata/"
    samples = [entry for entry in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, entry))]

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
        sample = random.choice(samples)
        parent_path = os.path.join(top_dir, sample)

        rgb_path = os.path.join(parent_path, f"image/{sample}.jpg")
        depth_path = os.path.join(parent_path, f"depth/{sample}_abs.png")
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path)

        intrinsics_path = os.path.join(parent_path, f"intrinsics.txt")

        extrinsics_dir_path = os.path.join(parent_path, f"extrinsics/")
        extrinsics_text_files = [f for f in os.listdir(extrinsics_dir_path) if f.endswith('.txt')]
        extrinsics_file = os.path.join(extrinsics_dir_path, extrinsics_text_files[0])

        ext_mat = np.genfromtxt(extrinsics_file, delimiter=" ")
        print(f"ext_mat= \n{ext_mat}")

        intrinsics = np.genfromtxt(intrinsics_path, delimiter=" ")

        print(f"intrinsics=\n{intrinsics}")


        K = intrinsic_obj(intrinsics, rgb_image.shape[1], rgb_image.shape[0])

        pose = homog_coord_to_pose_vector(ext_mat)

        depth_scale = 1

        graph = get_graph(client, owl, sam, rgb_image, depth_image, pose, K, depth_scale)

        gm.add_graph(graph)
        inp = input("press q to quit: ")
        i+=1

    
    
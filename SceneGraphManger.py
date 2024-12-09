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
    from magpie_control import realsense_wrapper as real
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from magpie_perception.label_owlv2 import LabelOWLv2
    from magpie_control.ur5 import UR5_Interface as robot
    from control_scripts import goto_vec
    from APIKeys import API_KEY
    from openai import OpenAI
    from config import realSenseFPS, topview_vec

    

    myrobot = robot()
    print(f"starting robot from observation")
    myrobot.start()


    myrs = real.RealSense(fps=realSenseFPS)
    myrs.initConnection()

    label_vit = LabelOWLv2(topk=1, score_threshold=0.01, cpu_override=False)
    label_vit.model.eval()
    print(f"{label_vit.model.device=}")

    sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    print(f"{sam_predictor.model.device=}")

    client = OpenAI(
        api_key= API_KEY,
    )

    goto_vec(myrobot, topview_vec)

    gm = Graph_Manager()
    inp = "a"
    load = True
    i = 0
    while inp != "q":
        if load:
            with open(f"./data_collection/g{i}.pkl", "rb") as f:
                graph = pickle.load(f)
        else:
            graph = get_graph(client, label_vit, sam_predictor, myrs, myrobot)
        gm.add_graph(graph)
        inp = input("press q to quit: ")
        i+=1
    myrobot.stop()
    myrs.disconnect()

    
    
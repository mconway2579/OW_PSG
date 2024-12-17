from SceneGraphGeneration import get_graph, PointCloud, semantic_graph_from_json, point_clound_graph_from_json
from helper_functions import get_points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import networkx as nx
import pickle
from config import n_state_samples, node_match_thresh, edge_match_thresh
from gpt_states import get_state

from semantics import str_semantic_distance, node_replacement_cost, node_match_func, edge_replacement_cost, edge_match_func, find_semantic_matching

def get_MonteCarlo_state(client, rgb_img, user_prompt, pose, display = False):
    """
    for a single rgb image and user_prompt
    create n_state_samples jsons like 
        {
            objects:[A,B,C]
            object relationships: [(A, is on, B), (C, is on, A)]
        }
    from each of those state samples create a graph
    then find the graph edit distance for each graph to each other graph filling in a distance matrix

    graph edit distance has the cosin distance for text embeddings as the replacement cost
    graph edit distance has one for the cost of adding or rremoving a node or edge
    it is always cheaper to substitute nodes that have some kindof similarity

    return the state that has the minimum row sum across the distance matrix
    """


    states = []
    for i in range(n_state_samples):
        _, state_json, _, _ = get_state(client, rgb_img, user_prompt, pose=pose)
        states.append(state_json)
        #print(f"{state_json}\n") if display else None
    graphs = []
    for state in states:
        graphs.append(semantic_graph_from_json(state))

    distance_matrix = np.zeros((len(graphs), len(graphs)))
    acc = 0
    for i in range(0, len(graphs)):
        g1 = graphs[i]
        for j in range(i+1, len(graphs)):
            g2 = graphs[j]
            sub_add_cost = lambda a=None,b=None: 1 #return the cos distance if nodes were not similar at all
            ged = nx.graph_edit_distance(g1, g2, node_match=node_match_func, edge_match=edge_match_func,
                                         node_subst_cost=node_replacement_cost, edge_subst_cost=edge_replacement_cost,
                                         node_del_cost=sub_add_cost ,node_ins_cost=sub_add_cost,
                                         edge_del_cost=sub_add_cost, edge_ins_cost=sub_add_cost,
                                         timeout = 120)
            distance_matrix[i,j] = ged
            acc +=1
            print(f"forming distance matrix {(i*len(graphs)) + j+1}/{len(graphs)**2}", end="\r") if display else None

    distance_matrix += distance_matrix.T

    print(f"distance_matrix=\n{distance_matrix}") if display else None
    distance_sums = np.sum(distance_matrix, axis=1)
    print(f"distance_sums=\n{distance_sums}") if display else None
    closest_idx = np.argmin(distance_sums)
    closest_state = states[closest_idx]
    return closest_state

def combine_nodes(G_out, G1, G2, g1_2_g2 = None, g2_2_g1 = None):
    """
    Combines nodes from G1 and G2 into G_out
    g1_2_g2: dict that matches a node str in g1 to a node str in g2
    g2_2_g1: inverse of g1_2_g2

    handles three cases
    1. a node exists in both g1 and g2 (has a matching in the two dictionaries)
    2. a node exists only in g1 but not in g2
    3. a node exists only in g2 but not in g1
    each case only adds a single node to G_out
    """

    print(f"\n\nBegin Combine Nodes")
    
    if g1_2_g2 is None or g2_2_g1 is None:
        g1_2_g2, g2_2_g1 = find_semantic_matching(list(G1.nodes), list(G2.nodes), node_match_thresh)

    G1_placed_nodes = []
    G2_placed_nodes = []

    #add nodes in both graphs
    for Name_1, Name_2 in g1_2_g2.items():
        Node_1 = G1.nodes[Name_1]
        Node_2 = G2.nodes[Name_2]

        new_node_obj = Node_1['data'] + Node_2['data']
        G_out.add_node(new_node_obj.str_label, data=new_node_obj, name=new_node_obj.str_label)

        G1_placed_nodes.append(Node_1["name"])
        G2_placed_nodes.append(Node_2["name"])
        print(f"G1({Name_1}) <-> G2({Name_2})")

    #add nodes in G1 but not G2
    for node, attr in G1.nodes(data=True):
        if attr['name'] not in G1_placed_nodes:
            G_out.add_node(attr["name"], data=attr["data"], name=attr['name'])
            G1_placed_nodes.append(attr["name"])
            print(f"{attr['name']} is only in G1")

    #add nodes in G2 but not G1
    for node, attr in G2.nodes(data=True):
        if attr['name'] not in G2_placed_nodes:
            G_out.add_node(attr["name"], data=attr["data"], name=attr['name'])
            G2_placed_nodes.append(attr["name"])
            print(f"{attr['name']} is only in G2")


    return G_out

def combine_edges(G_out, G1, G2, g1_to_g2 = None, g2_to_g1 = None):
    """
    Combines edges from G1 and G2 into G_out
    g1_2_g2: dict that matches a node str in g1 to a node str in g2
    g2_2_g1: inverse of g1_2_g2

    handles three cases
    1. a node exists in both g1 and g2, and the edge exists in both g1 and g2
    2. a node exists in both g1 and g2, but the edge only exists in g1
    3. a node exists in both g1 and g2, but the edge only exists in g2
    4. A node only exists in g1
    5. A node only exists in g2

    each case only adds a single edge to G_out
    """
    print(f"\n\nBegin Combine Edges")
    if g1_to_g2 is None or g2_to_g1 is None:
        g1_to_g2, g2_to_g1 = find_semantic_matching(list(G1.nodes), list(G2.nodes), node_match_thresh)

    G1_processed_nodes = []
    G2_processed_nodes = []

    #add edges for nodes in both G1 and G2
    #cases 1,2,3
    for Node1, Node2 in g1_to_g2.items():
        
        Node_1_edges = list(G1[Node1].items())
        Node_1_edges = [f"{edge[1]['name']}-{edge[0]}" for edge in Node_1_edges]

        Node_2_edges = list(G2[Node2].items())
        Node_2_edges = [f"{edge[1]['name']}-{edge[0]}" for edge in Node_2_edges]

        #print(f"{Node_1_edges=}")
        #print(f"{Node_2_edges=}")

        node1_placed_edges = []
        node2_placed_edges = []
        edgesNode1_2_edgesNode2, edgesNode2_2_edgesNode1= find_semantic_matching(Node_1_edges, Node_2_edges, edge_match_thresh)
        print(f"{edgesNode1_2_edgesNode2=}")

        #add edges in both node1 and node2, case1
        for e1, e2 in edgesNode1_2_edgesNode2.items():
            label, dest = e1.split("-")
            source = Node1
            G_out.add_edge(source, dest, name=label)

            node1_placed_edges.append(e1)
            node2_placed_edges.append(e2)

            print(f"{Node1}({e1}) <-> {Node2}({e2})")

        #add edges for node1 and node2 that are in only g1, case 2
        for edge in Node_1_edges:
            if edge not in node1_placed_edges:
                label, dest = edge.split("-")
                source = Node1
                G_out.add_edge(source, dest, name=label)
                node1_placed_edges.append(edge)
                print(f"{Node1} is a shared node but only has ({edge}) in G1")

        #add edges for node1 and node2 that are in only g2, case 3
        for edge in Node_2_edges:
            if edge not in node2_placed_edges:
                label, dest = edge.split("-")
                
                if dest in g2_to_g1.keys():
                    dest = g2_to_g1[dest]

                source = Node1
                G_out.add_edge(source, dest, name=label)
                node2_placed_edges.append(edge)
                print(f"{Node1} is a shared node but only has ({edge}) in G2")

        G1_processed_nodes.append(Node1)
        G2_processed_nodes.append(Node2)

    #add edges for nodes in G1 but not G2 case 4
    for Node1, attr in G1.nodes(data = True):
        if attr['name'] not in G1_processed_nodes:
            for neighbor in G1.neighbors(Node1):
                label = G1.get_edge_data(Node1, neighbor)["name"]
                G_out.add_edge(Node1, neighbor, name=label)
                print(f"G1 {Node1} ---{label}---> {neighbor}")
            G1_processed_nodes.append(attr["name"])


    #add edges for nodes in G2 but not G1 case 5
    for Node2, attr in G2.nodes(data = True):
        if attr['name'] not in G2_processed_nodes:
            for neighbor in G2.neighbors(Node2):
                #print(f"{neighbor=}")
                label = G2.get_edge_data(Node2, neighbor)["name"]
                neighbor_ref = neighbor
                if neighbor_ref in g2_to_g1.keys():
                    neighbor_ref = g2_to_g1[neighbor_ref]
                G_out.add_edge(Node2, neighbor_ref, name=label)
                print(f"G2 {Node2} ---{label}---> {neighbor_ref}")
            G2_processed_nodes.append(attr["name"])
    
    return G_out

def combine_graphs(G1, G2, display= True):
    G = nx.DiGraph()

    ts = None
    rgb = None
    depth = None
    if G1.graph["timestamp"] > G2.graph["timestamp"]:
        ts = G1.graph["timestamp"]
        rgb = G1.graph["rgb_img"]
        depth = G1.graph["depth_img"]
    else:
        ts = G2.graph["timestamp"]
        rgb = G2.graph["rgb_img"]
        depth = G2.graph["depth_img"]

    G.graph["timestamp"] = ts
    G.graph["rgb_img"] = rgb
    G.graph["depth_img"] = depth


    g1_2_g2, g2_to_g1 = find_semantic_matching(list(G1.nodes), list(G2.nodes), node_match_thresh)

    G = combine_nodes(G, G1, G2, g1_2_g2, g2_to_g1)
    G = combine_edges(G, G1, G2, g1_2_g2, g2_to_g1)

    return G

class Graph_Manager:
    """
    Class to manage adding graphs, combinging graphs, holding a graph history, and updating the display
    """
    def __init__(self):
        self.graph_history = []
        self.last_graphed_i = 0
        self.processed_graph = None

        self.fig = plt.figure(figsize = (19,12))
        self.ax2d_graph = self.fig.add_subplot(211)
        self.ax3d = self.fig.add_subplot(212, projection='3d')
        
        self.start()
        
    def update_display(self, rubbish):
        if len(self.graph_history) == 0:
            return
        if len(self.graph_history) == self.last_graphed_i:
            return
        G = self.processed_graph
        self.ax2d_graph.clear()
        try:
            pos = nx.planar_layout(G)
        except nx.NetworkXException as e:
            print(f"Error: {e}")
            pos = nx.spring_layout(G)
        # Draw the graph on the specified axes
        nx.draw(G, pos=pos, ax=self.ax2d_graph, with_labels=True, node_color='lightblue', node_size=1500, font_size=15)

        # Draw edge labels on the same axes
        edge_labels = nx.get_edge_attributes(G, 'name')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=12, ax=self.ax2d_graph)

        points, colors = get_points(G)

        self.ax3d.clear()
        self.ax3d.set_xlim(-0.5, 0)
        self.ax3d.set_ylim(-1, 0)
        self.ax3d.set_zlim(0, 0.2)
        self.ax3d.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=0.5)

        plt.tight_layout()

        self.last_graphed_i = len(self.graph_history)

    def add_graph(self, client, owl, sam, rgb_img, depth_img, pose, K, depth_scale, user_prompt, display = False):
        state = get_MonteCarlo_state(client, rgb_img, user_prompt, pose=pose, display = True)
        print(f"\n{len(self.graph_history)}: {state}")
        PC_G = point_clound_graph_from_json(state, rgb_img, depth_img, pose, owl, sam, K, depth_scale, display = display)
        
        self.graph_history.append(PC_G)
        if self.processed_graph is not None:
            self.processed_graph = combine_graphs(self.processed_graph, PC_G)
        else:
            self.processed_graph = PC_G
        return PC_G

    def start(self):
        self.animation = FuncAnimation(self.fig, self.update_display, interval=1000, cache_frame_data=False)  # 1 Hz update
        plt.show(block = False)
        plt.pause(1)


if __name__ == "__main__":
    from APIKeys import API_KEY
    import os
    from SceneGraphGeneration import intrinsic_obj, OWLv2, SAM2
    from openai import OpenAI
    

    top_dir = f"./custom_dataset/one on two/"
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
    for i in range(len(samples)):
        sample = samples[i]
        with open(sample, "rb") as file:
            rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
            K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])
        
        gm.add_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale, "how are objects layed out on the table? I know some of the objects are blocks", display=True)
        #input("press enter to continue")
    input("press enter to quit:\n")
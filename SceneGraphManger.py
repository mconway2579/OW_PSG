from SceneGraphGeneration import get_graph, Node, semantic_graph_from_json, point_clound_graph_from_json
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

from semantics import str_semantic_distance, node_replacement_cost, node_match_func, edge_replacement_cost, edge_match_func

def get_MonteCarlo_state(client, rgb_img, user_prompt, pose, display = False):
    states = []
    for i in range(n_state_samples):
        _, state_json, _, _ = get_state(client, rgb_img, user_prompt, pose=pose, display = False)
        states.append(state_json)
        print(f"{state_json}\n") if display else None
    graphs = []
    for state in states:
        graphs.append(semantic_graph_from_json(state))

    distance_matrix = np.zeros((len(graphs), len(graphs)))
    acc = 0
    for i, g1 in enumerate(graphs):
        for j, g2 in enumerate(graphs):
            if g1 == g2:
                continue
            sub_add_cost = lambda a=None,b=None: 1 #return the cos distance if nodes were not similar at all
            ged = nx.graph_edit_distance(g1, g2, node_match=node_match_func, edge_match=edge_match_func,
                                         node_subst_cost=node_replacement_cost, edge_subst_cost=edge_replacement_cost,
                                         node_del_cost=sub_add_cost ,node_ins_cost=sub_add_cost,
                                         edge_del_cost=sub_add_cost, edge_ins_cost=sub_add_cost,
                                         timeout = 120)
            distance_matrix[i,j] = ged
            acc +=1
            print(f"forming distance matrix {(i*len(graphs)) + j+1}/{len(graphs)**2}", end="\r") if display else None

    print(f"distance_matrix=\n{distance_matrix}") if display else None
    distance_sums = np.sum(distance_matrix, axis=1)
    print(f"distance_sums=\n{distance_sums}") if display else None
    closest_idx = np.argmin(distance_sums)
    closest_state = states[closest_idx]
    return closest_state

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

def combine_nodes(G_out, G1, G2, g1_2_g2 = None, g2_to_g1 = None):
    print(f"\n\nBegin Combine Nodes")
    
    if g1_2_g2 is None or g2_to_g1 is None:
        g1_2_g2, g2_to_g1 = find_semantic_matching(list(G1.nodes), list(G2.nodes), node_match_thresh)

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
    print(f"\n\nBegin Combine Edges")
    if g1_to_g2 is None or g2_to_g1 is None:
        g1_to_g2, g2_to_g1 = find_semantic_matching(list(G1.nodes), list(G2.nodes), node_match_thresh)

    G1_processed_nodes = []
    G2_processed_nodes = []

    #add edges for nodes in both G1 and G2
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
        #add edges in both node1 and node2
        for e1, e2 in edgesNode1_2_edgesNode2.items():
            label, dest = e1.split("-")
            source = Node1
            G_out.add_edge(source, dest, name=label)

            node1_placed_edges.append(e1)
            node2_placed_edges.append(e2)

            print(f"{Node1}({e1}) <-> {Node2}({e2})")

        #add edges for node1 and node2 that are in only g1
        for edge in Node_1_edges:
            if edge not in node1_placed_edges:
                label, dest = edge.split("-")
                source = Node1
                G_out.add_edge(source, dest, name=label)
                node1_placed_edges.append(edge)
                print(f"{Node1} is a shared node but only has ({edge}) in G1")

        #add edges for node1 and node2 that are in only g2
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

    #add edges for nodes in G1 but not G2
    for Node1, attr in G1.nodes(data = True):
        if attr['name'] not in G1_processed_nodes:
            for neighbor in G1.neighbors(Node1):
                label = G1.get_edge_data(Node1, neighbor)["name"]
                G_out.add_edge(Node1, neighbor, name=label)
                print(f"G1 {Node1} ---{label}---> {neighbor}")
            G1_processed_nodes.append(attr["name"])


    #add edges for nodes in G2 but not G1
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

    G = combine_nodes(G, G1, G2)
    G = combine_edges(G, G1, G2)

    return G

class Graph_Manager:
    def __init__(self):
        self.graph_history = []
        self.processed_graph = None

        self.fig = plt.figure(figsize = (19,12))
        self.ax2d_graph = self.fig.add_subplot(211)
        self.ax3d = self.fig.add_subplot(212, projection='3d')
        
        self.start()
        
    def update_display(self, rubbish):
        if len(self.graph_history) == 0:
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
        state = get_MonteCarlo_state(client, rgb_img, user_prompt, pose=pose, display = False)
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
        input("press enter to continue")
    input("press enter to quit:\n")
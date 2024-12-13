import open3d as o3d
from helper_functions import pose_vector_to_homog_coord, homog_coord_to_pose_vector, display_graph, draw_bounding_boxes
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
import networkx as nx
from gpt_states import get_state
import time
import pickle
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from config import vit_model_name, vit_thresh
from sam2.sam2_image_predictor import SAM2ImagePredictor


class intrinsic_obj:
    def __init__(self, array, width, height):
        if array.shape == (3,3):
            array = array.flatten()
        #expects array like [518.858   0.    284.582   0.    519.47  208.736   0.      0.      1.   ]
        #fills in K.width, K.height, K.fx, K.fy, K.ppx, K.ppy
        self.fx = array[0]
        self.ppx = array[2]
        self.fy = array[4]
        self.ppy = array[5]
        self.width = width
        self.height = height

class OWLv2:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained(vit_model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(vit_model_name)

        self.model.to(torch.device("cuda")) if torch.cuda.is_available() else None
        self.model.to(torch.device("mps")) if torch.backends.mps.is_available() else None
        self.model.eval()  # set model to evaluation mode
    def predict(self, img, querries, k = 1):
        inputs = self.processor(text=querries, images=img, return_tensors="pt")
        inputs.to(torch.device("cuda")) if torch.cuda.is_available() else None
        inputs.to(torch.device("mps")) if torch.backends.mps.is_available() else None

        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([img.shape[:2]])  # (height, width)

        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)[0]
        #print(f"\n\n{results}\n\n")
        scores = results["scores"]
        labels = results["labels"]
        boxes = results["boxes"]
        unique_classes = torch.unique(labels)

        highest_score_boxes = []

        # Find the highest score box for each class
        for cls in unique_classes:
            # Get indices of the current class
            class_indices = (labels == cls).nonzero(as_tuple=True)[0]
            
            # Get scores for the current class
            class_scores = scores[class_indices]
            
            # Find the index of the maximum score
            max_index = class_indices[torch.argmax(class_scores)]
            
            # Get the corresponding box and score
            highest_score_boxes.append((querries[int(cls)], boxes[max_index].tolist()))
        return highest_score_boxes

    def __str__(self):
        return f"OWLv2: {self.model.device}"
    def __repr__(self):
        return self.__str__()

class SAM2:
    def __init__(self):
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    def predict(self, img, bbox):
        # Suppress warnings during the prediction step
        self.sam_predictor.set_image(img)

        sam_mask = None
        sam_scores = None
        sam_logits = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sam_mask, sam_scores, sam_logits = self.sam_predictor.predict(box=bbox)

        #print(f"{sam_mask=}")
        #print(f"{type(sam_mask)=}")
        #print(f"{dir(sam_mask)=}")
        #print(f"{sam_mask.shape=}")

        
        sam_mask = np.all(sam_mask, axis=1)
        #print(f"{sam_mask.shape=}")
        return sam_mask, sam_scores, sam_logits
    def __str__(self):
        return f"SAM2: {self.sam_predictor.model.device}"
    def __repr__(self):
        return self.__str__()

class Node:
    def __init__(self, str_label, points, colors):
        self.voxel_size = 0.005  # adjust based on your data

        self.str_label = str_label

        self.points = points
        self.colors = colors
        #self.clean_pointcloud()

    def clean_pointcloud(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        self.points = np.asarray(pcd.points)
        self.colors = np.asarray(pcd.colors)

    def display(self, blocking = False):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], c=self.colors, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"{self.str_label} Point Cloud")
        plt.show(block = blocking)
        if not blocking:
            plt.pause(1)

    def __add__(self, other):
        assert isinstance(other, Node), "Can only add Node instances"
        self.points = np.concatenate((self.points, other.points), axis=0)
        self.colors = np.concatenate((self.colors, other.colors), axis=0)
        self.clean_pointcloud()
        return self

def semantic_graph_from_json(state_json, display = False):
    G = nx.DiGraph()
    for obj_str in state_json['objects']:
        G.add_node(obj_str, name=obj_str)
    for relation in state_json['object_relationships']:
        G.add_edge(relation[0], relation[2], connection=relation[1])
    if display:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        # Draw the graph on the specified axes
        nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=15)
        # Draw edge labels on the same axes
        edge_labels = nx.get_edge_attributes(G, 'connection')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=12)
        # Show without blocking, then pause
        plt.tight_layout()
        plt.show(block = False)
        plt.pause(1)
    return G

def point_clound_graph_from_json(state_json, rgb_img, depth_img, pose, label_vit, sam_predictor, K, depth_scale, display = False):
    G = nx.DiGraph()
    G.graph["timestamp"] = time.time()
    G.graph["observation_pose"] = pose
    G.graph["rgb_img"] = rgb_img
    G.graph["depth_img"] = depth_img
    labels_bboxes_list = label_vit.predict(rgb_img, state_json["objects"])
    bboxes = [bbox for _, bbox in labels_bboxes_list]
    labels = [label for label, _ in labels_bboxes_list]
    if display:
        draw_bounding_boxes(rgb_img, bboxes, labels)

    #print(f"{bboxes=}")
    #print(f"{labels=}")

    masks, scores, logits = sam_predictor.predict(rgb_img, bboxes)

    for label, mask in zip(labels, masks):
        rgb_segment = rgb_img.copy()
        rgb_segment[~mask] = 0
        depth_segment = depth_img.copy()
        depth_segment[~mask] = 0

        if display:
            plt.figure(figsize=(12, 12))
            plt.imshow(rgb_segment)
            plt.title(f"{label} Mask")
            plt.show(block=False)
            plt.pause(1)

        temp_rgb_img = o3d.geometry.Image(rgb_segment)
        temp_depth_img = o3d.geometry.Image(depth_segment)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(temp_rgb_img, temp_depth_img, depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.ppx, K.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        transform_matrix = pose_vector_to_homog_coord(pose)
        pcd.transform(transform_matrix)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        obj_node = Node(label, points, colors)
        G.add_node(label, data=obj_node, name=label)

    for edge in state_json["object_relationships"]:
        G.add_edge(edge[0], edge[2], connection=edge[1])
        
    return G

def get_graph(OAI_Client, label_vit, sam_predictor, rgb_img, depth_img, pose, K, depth_scale, prompt):
    _, state_json, _, _ = get_state(OAI_Client, rgb_img, prompt)
    print(state_json)
    G = point_clound_graph_from_json(state_json, rgb_img, depth_img, pose, label_vit, sam_predictor, K, depth_scale)
    return G

if __name__ == "__main__":
    from APIKeys import API_KEY
    from openai import OpenAI
    import os
    import cv2
    
    with open("./custom_dataset/one on two/top_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)

    K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])

    sam = SAM2()
    print(f"{sam=}")
    owl = OWLv2()
    print(f"{owl=}")
    

    client = OpenAI(
        api_key= API_KEY,
    )
    prompt = "how are objects layed out on the table?"
    graph = get_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale, prompt)
    if False:
        for obj, node in graph.nodes(data=True):
            try:
                node["data"].display()
            except KeyError:
                print(f"Key error retriving data from {obj}, {node}")
    display_graph(graph, blocking = True)

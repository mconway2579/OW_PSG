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
from config import vit_model_name, voxel_size
from sam2.sam2_image_predictor import SAM2ImagePredictor

#Class to mirror the realsense intrinsic object
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
#Class to use OWLv2
class OWLv2:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained(vit_model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(vit_model_name)

        self.model.to(torch.device("cuda")) if torch.cuda.is_available() else None
        self.model.to(torch.device("mps")) if torch.backends.mps.is_available() else None
        self.model.eval()  # set model to evaluation mode
    def predict(self, img, querries):
        """
        Gets realsense frames
        Parameters:
        - img: image to produce bounding boxes in
        - querries: list of strings whos bounding boxes we want

        Returns:
        - highest_score_boxes: list of bounding boxes associated with querries
        """
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

#Class to use sam2
class SAM2:
    def __init__(self):
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    def predict(self, img, bbox):
        """
        Gets realsense frames
        Parameters:
        - img: image to produce masks in in
        - bbox: list of bounding boxes whos masks we want

        Returns:
        - sam_mask: masks produced by sam for every bounding box
        - sam_scores: scores produced by sam for every mask
        - sam_logits: logits produced by sam for every mask
        """
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


#PointCloud class stored as data attribute in an nx graph
class PointCloud:
    def __init__(self, str_label, points, colors):
        """
        A Point Cloud has a string label
        points in 3d space in the world frame
        colors for each point
        """
        self.str_label = str_label
        self.points = points
        self.colors = colors
        self.clean_pointcloud()

    def clean_pointcloud(self):
        """
        casts nodes points and colors into an o3d point could,
        downsamples with voxel size from config
        removes statistical outliers
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        #pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
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
        assert isinstance(other, PointCloud), "Can only add point cloud instances"
        points = np.concatenate((self.points, other.points), axis=0)
        colors = np.concatenate((self.colors, other.colors), axis=0)
        #self.clean_pointcloud()
        return PointCloud(self.str_label, points, colors)

def semantic_graph_from_json(state_json, display = False):
    """
    Given a json like 
    {
        objects:[A,B,C]
        object relationships: [(A, is on, B), (C, is on, A)]
    }
    Create an nx graph where nodes have string names and edges have string names
    """
    G = nx.DiGraph()
    nodes = state_json["objects"]
    for edge in state_json["object_relationships"]:
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[2] not in nodes:
            nodes.append(edge[2])
    
    for obj_str in nodes:
        G.add_node(obj_str, name=obj_str)

    for relation in state_json['object_relationships']:
        assert relation[0] in G.nodes, f"{relation[0]} not in {G.nodes=}"
        assert relation[2] in G.nodes, f"{relation[2]} not in {G.nodes=}"
        G.add_edge(relation[0], relation[2], name=relation[1])

    if display:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        # Draw the graph on the specified axes
        nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=15)
        # Draw edge labels on the same axes
        edge_labels = nx.get_edge_attributes(G, 'name')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=12)
        # Show without blocking, then pause
        plt.tight_layout()
        plt.show(block = False)
        plt.pause(1)
    return G

def point_clound_graph_from_json(state_json, rgb_img, depth_img, pose, label_vit, sam_predictor, K, depth_scale, display = False):
    """
    Given a json like 
    {
        objects:[A,B,C]
        object relationships: [(A, is on, B), (C, is on, A)]
    }
    Create an nx graph where nodes have attached point cloud objects
    """
    G = nx.DiGraph()
    G.graph["timestamp"] = time.time()
    G.graph["observation_pose"] = pose
    G.graph["rgb_img"] = rgb_img
    G.graph["depth_img"] = depth_img

    #verify no objects are in object relatonships but not in objects
    nodes = state_json["objects"]
    for edge in state_json["object_relationships"]:
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[2] not in nodes:
            nodes.append(edge[2])

    #get bounding boxes for each node
    oneshot = True
    labels = []
    bboxes = []
    if oneshot:
        labels_bboxes_list = label_vit.predict(rgb_img, nodes)
        bboxes = [bbox for _, bbox in labels_bboxes_list]
        labels = [label for label, _ in labels_bboxes_list]
    else:
        for node in nodes:
            labels_bboxxes_list = label_vit.predict(rgb_img, [node])
            labels.append(node)
            bboxes.append(labels_bboxxes_list[0][1])
    
    if display:
        draw_bounding_boxes(rgb_img, bboxes, labels)
    #get masks for each object
    masks, scores, logits = sam_predictor.predict(rgb_img, bboxes)

    for label, mask in zip(labels, masks):
        #create a point cloud for each mask label pair and store it as a node
        rgb_segment = rgb_img.copy()
        rgb_segment[~mask] = 0
        depth_segment = depth_img.copy()
        depth_segment[~mask] = 0

        temp_rgb_img = o3d.geometry.Image(rgb_segment)
        temp_depth_img = o3d.geometry.Image(depth_segment)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(temp_rgb_img, temp_depth_img, depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.ppx, K.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        transform_matrix = pose_vector_to_homog_coord(pose)
        pcd.transform(transform_matrix)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        obj_node = PointCloud(label, points, colors)
        G.add_node(label, data=obj_node, name=label)

        if display:
            fig = plt.figure(figsize=(12, 12))
            mask_ax = fig.add_subplot(2, 2, 1)
            mask_ax.set_title("SAM2 Mask")

            rgbSeg_ax = fig.add_subplot(2, 2, 2)
            rgbSeg_ax.set_title("RGB segment")

            depthSeg_ax = fig.add_subplot(2, 2, 3)
            depthSeg_ax.set_title("Depth segment")

            pc_ax = fig.add_subplot(2, 2, 4, projection='3d')  # 3D subplot
            pc_ax.set_title("Point Cloud")


            mask_ax.imshow(mask)
            rgbSeg_ax.imshow(rgb_segment)
            depthSeg_ax.imshow(depth_segment)
            pc_ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=1)
            pc_ax.set_xlabel("X")
            pc_ax.set_ylabel("Y")
            pc_ax.set_zlabel("Z")
        #print(f"added_node {label=} {obj_node=}")

    for edge in state_json["object_relationships"]:
        assert edge[0] in G.nodes, f"{edge[0]} not in {G.nodes=}"
        assert edge[2] in G.nodes, f"{edge[2]} not in {G.nodes=}"
        G.add_edge(edge[0], edge[2], name=edge[1])
        
    return G

def get_graph(OAI_Client, label_vit, sam_predictor, rgb_img, depth_img, pose, K, depth_scale, prompt):
    #gets a state from openai
    _, state_json, _, _ = get_state(OAI_Client, rgb_img, prompt, pose=pose)
    print(state_json)
    #converts state into pointcloud graph
    G = point_clound_graph_from_json(state_json, rgb_img, depth_img, pose, label_vit, sam_predictor, K, depth_scale, display=True)
    return G

if __name__ == "__main__":
    from APIKeys import API_KEY
    from openai import OpenAI

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
            node["data"].display()
    
            
    display_graph(graph, blocking = True)

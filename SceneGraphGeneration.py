import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
import warnings
from magpie_control.ur5 import pose_vector_to_homog_coord, homog_coord_to_pose_vector
import networkx as nx
from gpt_states import get_state
import time
import pickle

def get_observation_patch(obs, edge_color = "r"):
    rect = patches.Rectangle(
                            (obs.pix_xmin, obs.pix_ymin),
                                obs.pix_xmax - obs.pix_xmin,
                                obs.pix_ymax - obs.pix_ymin,
                                linewidth=2, edgecolor=edge_color, facecolor='none'
                            )
    return rect

class Node:
    def __init__(self, str_label, rgb_img, depth_img, label_vit, sam_predictor, K, depth_scale, observation_pose):
        self.str_label = str_label

        self.pix_xmin = None
        self.pix_xmax = None
        self.pix_ymin = None
        self.pix_ymax = None
        
        self.mask = None
        
        self.rgb_segment = None
        self.depth_segment = None

        self.points = None
        self.colors = None


        self.OWLv2_logits = None
        self.sam_logits = None

        self.calc_bbox(rgb_img, label_vit)
        self.calc_pcd(rgb_img, depth_img, K, depth_scale, observation_pose, sam_predictor)
    def calc_bbox(self, rgb_img, label_vit):
        bbox = None
        with torch.no_grad():
            bbox = label_vit.label(rgb_img, self.str_label, self.str_label, plot=False, topk=True)
            bbox = bbox[1][0].tolist()
        self.pix_xmin = int(bbox[0])
        self.pix_ymin = int(bbox[1])
        self.pix_xmax = int(bbox[2])
        self.pix_ymax = int(bbox[3])

    def calc_pcd(self, rgb_img, depth_img, K, depth_scale, observation_pose, sam_predictor):
        sam_predictor.set_image(rgb_img)
        sam_box = np.array([self.pix_xmin,  self.pix_ymin,  self.pix_xmax,  self.pix_ymax])


        # Suppress warnings during the prediction step
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            sam_mask, sam_scores, sam_logits = sam_predictor.predict(box=sam_box)



        self.sam_logits = sam_logits
        sam_mask = np.all(sam_mask, axis=0)
        #expanded_sam_mask = np.expand_dims(sam_mask, axis=-1)
        
        self.mask = sam_mask
        self.rgb_segment = rgb_img.copy()
        self.rgb_segment[~sam_mask] = 0
        self.depth_segment = depth_img.copy()
        self.depth_segment[~sam_mask] = 0

        temp_rgb_img = o3d.geometry.Image(self.rgb_segment)
        temp_depth_img = o3d.geometry.Image(self.depth_segment)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(temp_rgb_img, temp_depth_img, depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)
        #print(f"{dir(K)=}")
        intrinsic = o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.ppx, K.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        #pcd = pcd.uniform_down_sample(every_k_points=5)
        #pcd = pcd.voxel_down_sample(voxel_size=0.001)  # Down-sample with finer detail
        transform_matrix = pose_vector_to_homog_coord(observation_pose)
        pcd.transform(transform_matrix)
        
     
        #self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=1.0)
        #pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
        voxel_size = 0.001  # adjust based on your data
        #pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        #self.pcd_bbox = self.pcd.get_axis_aligned_bounding_box()
        #self.pcd_bbox = pcd.get_minimal_oriented_bounding_box()
        #self.pcd_bbox.color = (1,0,0)
        self.points = np.asarray(pcd.points)
        self.colors = np.asarray(pcd.colors)
        
    def display(self, blocking = False):
        fig, axes = plt.subplots(ncols=2, nrows=2)
        axes[0, 0].imshow(self.rgb_segment)
        axes[0, 0].add_patch(get_observation_patch(self, "r"))
        axes[0, 0].text(self.pix_xmin, self.pix_ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')
        axes[0, 0].set_title("RGB segment")

        axes[0, 1].imshow(self.depth_segment)
        axes[0, 1].add_patch(get_observation_patch(self, "r"))
        axes[0, 1].text(self.pix_xmin, self.pix_ymin - 10, f"{self.str_label}", color='r', fontsize=12, ha='left', va='bottom')
        axes[0, 1].set_title("Depth segment")
    
        plt.tight_layout()
        #print(f"Showing observation for {self.str_label}")
        plt.show(block = False)
        plt.pause(1)  # Keeps the figure open for 3 seconds
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], c=self.colors, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"{self.str_label} Point Cloud")
        plt.show(block = blocking)
        if not blocking:
            plt.pause(1)
 
def get_graph(OAI_Client, label_vit, sam_predictor, rgb_img, depth_img, pose, K, depth_scale):
    G = nx.DiGraph()

    G.graph["timestamp"] = time.time()
    G.graph["observation_pose"] = pose
    

    _, state_json, _, _ = get_state(OAI_Client, rgb_img)
    print(state_json)

    for object in state_json["objects"]:
        obj_node = Node(object, rgb_img, depth_img, label_vit, sam_predictor, K, depth_scale, pose)
        G.add_node(object, data=obj_node)

    for edge in state_json["object_relationships"]:
        G.add_edge(edge[0], edge[2], connection=edge[1])
        
    return G

def get_points(G):
    points = []
    colors = []

    for obj, node in G.nodes(data=True):
        #print(f"{obj=}")
        #print(f"{node=}")
        try:
            points.append(node["data"].points)
            colors.append(node["data"].colors)
        except KeyError:
            print(f"Key error retriving data from {obj}, {node}")
        
    
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    return points, colors

def display_graph(G, blocking = False):
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 12))
    points, colors = get_points(G)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pos = nx.spring_layout(G)
    # Draw the graph on the specified axes
    nx.draw(G, pos=pos, ax=axes[0], with_labels=True, node_color='lightblue', node_size=1500, font_size=15)
    # Draw edge labels on the same axes
    edge_labels = nx.get_edge_attributes(G, 'connection')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=12, ax=axes[0])
    # Show without blocking, then pause
    plt.tight_layout()
    plt.show(block=blocking)
    if not blocking:
        plt.pause(1)


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
if __name__ == "__main__":
    from APIKeys import API_KEY
    import os
    import cv2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from magpie_perception.label_owlv2 import LabelOWLv2
    from openai import OpenAI


    sample = "img_0847"
    parent_path = f"/home/max/OW_PSG/SUNRGBD/kv1/b3dodata/{sample}/"

    rgb_path = os.path.join(parent_path, f"image/{sample}.jpg")
    depth_path = os.path.join(parent_path, f"depth/{sample}_abs.png")
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path)

    intrinsics_path = os.path.join(parent_path, f"intrinsics.txt")

    extrinsics_dir_path = os.path.join(parent_path, f"extrinsics/")
    extrinsics_text_files = [f for f in os.listdir(extrinsics_dir_path) if f.endswith('.txt')]
    extrinsics_file = os.path.join(extrinsics_dir_path, extrinsics_text_files[0])

    ext_mat = np.genfromtxt(extrinsics_file, delimiter=" ")
    #print(f"ext_mat= \n{ext_mat}")

    intrinsics = np.genfromtxt(intrinsics_path, delimiter=" ")
    #print(f"intrinsics=\n{intrinsics}")
    #print(f"{intrinsics.shape}")


    K = intrinsic_obj(intrinsics, rgb_image.shape[1], rgb_image.shape[0])

    label_vit = LabelOWLv2(topk=1, score_threshold=0.01, cpu_override=False)
    label_vit.model.eval()
    print(f"{label_vit.model.device=}")

    sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    print(f"{sam_predictor.model.device=}")

    client = OpenAI(
        api_key= API_KEY,
    )

    pose = homog_coord_to_pose_vector(ext_mat)

    depth_scale = 1
    

    #obs = Node("chair", rgb_image, depth_image, label_vit, sam_predictor, K, depth_scale, pose)
    #obs.display()

    graph = get_graph(client, label_vit, sam_predictor, rgb_image, depth_image, pose, K, depth_scale)
    #for obj, node in graph.nodes(data=True):
    #    try:
    #        node["data"].display()
    #    except KeyError:
    #        print(f"Key error retriving data from {obj}, {node}")
    display_graph(graph, blocking = True)
    with open(f"./data_collection/g{sample}.pkl", "wb") as f:
        pickle.dump(graph, f)


    #myrobot.stop()
    #myrs.disconnect()
    
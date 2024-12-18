from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import pickle
import open3d as o3d
import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings


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

def pose_vector_to_homog_coord(pose):
    """
    Convert a 6-element pose vector [x, y, z, roll, pitch, yaw]
    into a 4x4 homogeneous transformation matrix.
    
    Parameters
    ----------
    pose : array_like of shape (6,)
        Pose given as (x, y, z, roll, pitch, yaw) in radians.
        
    Returns
    -------
    H : ndarray of shape (4,4)
        4x4 homogeneous transformation matrix.
    """
    x, y, z, roll, pitch, yaw = pose
    
    # Compute rotation matrix from roll, pitch, yaw
    # Assuming the rotation sequence: Rz(yaw)*Ry(pitch)*Rx(roll)
    # Note: This is a common convention, but verify with your system.
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    # Rotation about X (roll)
    Rx = np.array([[1,    0,     0   ],
                   [0,    cr,   -sr ],
                   [0,    sr,    cr ]])
    
    # Rotation about Y (pitch)
    Ry = np.array([[ cp,   0,   sp],
                   [  0,   1,    0],
                   [-sp,   0,   cp]])
    
    # Rotation about Z (yaw)
    Rz = np.array([[cy, -sy,  0],
                   [sy,  cy,  0],
                   [0,    0,   1]])
    
    # Combined rotation R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    
    # Construct homogeneous transformation matrix
    H = np.eye(4)
    H[0:3, 0:3] = R
    H[0:3, 3]   = [x, y, z]
    return H

#Class to use OWLv2
class OWLv2:
    def __init__(self):
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

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
OWL = OWLv2()
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

        
        sam_mask = np.all(sam_mask, axis=0)
        #print(f"{sam_mask.shape=}")
        return sam_mask, sam_scores, sam_logits
    def __str__(self):
        return f"SAM2: {self.sam_predictor.model.device}"
    def __repr__(self):
        return self.__str__()
SAM = SAM2()

def get_point_cloud(obj_str, rgb_img, d_img, K, d_scale, pose, title=None):
    labels_bboxes_list = OWL.predict(rgb_img, obj_str)
    bboxes = [bbox for _, bbox in labels_bboxes_list]
    #print(bboxes)
    labels = [label for label, _ in labels_bboxes_list]
    mask, scores, logits = SAM.predict(rgb_img, bboxes)
    #print(f"{mask.shape=}")
    #print(f"{rgb_img.shape=}")
    #print(f"{depth_img.shape=}")

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
    
    if title is not None:
        fig, axes = plt.subplots(nrows=2, ncols=1)
        axes[0].set_title(title)
        axes[0].imshow(rgb_segment)
        axes[1].imshow(depth_segment)
        plt.show(block=False)
        plt.pause(1)
    

    return points, colors

def display_pointcloud(points, colors, title=None, blocking=False):
    """
    Displays a 3D point cloud with color mapping.
    
    Args:
        points (array-like): Nx3 array of 3D points (x, y, z).
        colors (array-like): Nx3 array of RGB colors.
        title (str, optional): Title of the plot.
        blocking (bool, optional): If True, the plot will block execution.
    """

    # Clamp negative values to 0
    colors = np.clip(colors, 0, None)

    # Normalize colors to [0, 1]
    if colors.max() > 0:
        colors = colors / colors.max()

    # Debugging: Check colors after normalization
    print("Colors after normalization - Max:", colors.max(), "Min:", colors.min())
    
    # Validate shapes
    if points.shape[0] != colors.shape[0]:
        raise ValueError("Number of points and colors must match.")
    if colors.shape[1] != 3:
        raise ValueError("Colors should be a Nx3 array representing RGB values.")
    
    # Create the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)

    # Add labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    if title is not None:
        ax.set_title(title)
    
    # Show the plot
    plt.show(block=blocking)
    if not blocking:
        plt.pause(1)

if __name__ == "__main__":
    search_str = "yellow_block"
    points = None
    colors = None
    with open("./custom_dataset/one on two/top_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
    K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])
    tv_points, tv_colors = get_point_cloud(search_str, rgb_img, depth_img, K, depth_scale, pose, title="tv")
    tv_color = np.array([0, 0, 1])  # Shape: (1, 3)
    tv_colors = np.tile(tv_color, (tv_points.shape[0], 1))  # Shape: (N, 3)
    display_pointcloud(tv_points, tv_points, "topview point cloud")


    with open("./custom_dataset/one on two/side_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
    K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])
    sv_points, sv_colors = get_point_cloud(search_str, rgb_img, depth_img, K, depth_scale, pose, title="sv")
    sv_color = np.array([0, 1, 0])  # Shape: (1, 3)
    sv_colors = np.tile(sv_color, (sv_points.shape[0], 1))  # Shape: (N, 3)
    display_pointcloud(sv_points, sv_colors, "sideview point cloud")


    with open("./custom_dataset/one on two/angled_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
    K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])
    av_points, av_colors = get_point_cloud(search_str, rgb_img, depth_img, K, depth_scale, pose, title="av")
    av_color = np.array([1, 0, 0])  # Shape: (1, 3)
    av_colors = np.tile(av_color, (av_points.shape[0], 1))  # Shape: (N, 3)
    display_pointcloud(av_points, av_colors, "angled_view point cloud")


    with open("./custom_dataset/one on two/angled_view2.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)
    K = intrinsic_obj(K, rgb_img.shape[1], rgb_img.shape[0])
    av2_points, av2_colors = get_point_cloud(search_str, rgb_img, depth_img, K, depth_scale, pose, title="av2")
    av2_color = np.array([1, 1, 0])  # Shape: (1, 3)
    av2_colors = np.tile(av2_color, (av2_points.shape[0], 1))  # Shape: (N, 3)
    display_pointcloud(av2_points, av2_colors, "angled_view2 point cloud")



    combined_points = np.concatenate((tv_points, sv_points, av_points, av2_points))
    combined_colors = np.concatenate((tv_colors, sv_colors, av_colors, av2_colors))
    display_pointcloud(combined_points, combined_colors, "combined point cloud", blocking=True)





import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def homog_coord_to_pose_vector(H):
    """
    Convert a 4x4 homogeneous transformation matrix into a 
    6-element pose vector [x, y, z, roll, pitch, yaw].
    
    Parameters
    ----------
    H : ndarray of shape (4,4)
        Homogeneous transformation matrix.
        
    Returns
    -------
    pose : ndarray of shape (6,)
        Pose given as (x, y, z, roll, pitch, yaw) in radians.
    """
    # Extract the rotation part and the translation
    R = H[0:3, 0:3]
    x, y, z = H[0:3, 3]
    
    pitch = np.arcsin(-R[2,0])
    
    # Once we have pitch:
    # roll = atan2(R[2,1], R[2,2])
    roll = np.arctan2(R[2,1], R[2,2])
    
    # yaw = atan2(R[1,0], R[0,0])
    yaw = np.arctan2(R[1,0], R[0,0])
    
    return np.array([x, y, z, roll, pitch, yaw])

def get_points(G):
    points = []
    colors = []

    for obj, node in G.nodes(data=True):
        points.append(node["data"].points)
        colors.append(node["data"].colors)
        
        
    
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

def draw_bounding_boxes(rgb_image, bboxes, labels, label_colors=None):
    """
    Draw bounding boxes on an RGB image.
    
    Args:
        rgb_image (numpy.ndarray): The RGB image as a NumPy array.
        bboxes (list of lists): List of bounding boxes, where each box is [x_min, y_min, x_max, y_max].
        labels (list): List of labels corresponding to the bounding boxes.
        label_colors (dict, optional): A dictionary mapping labels to colors. Random colors will be generated if None.
    """
    # Generate random colors for labels if not provided
    if label_colors is None:
        unique_labels = set(labels)
        label_colors = {label: np.random.rand(3,) for label in unique_labels}

    # Create a matplotlib figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(rgb_image)

    # Add each bounding box to the image
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # Add a rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2,
            edgecolor=label_colors[label], facecolor='none'
        )
        ax.add_patch(rect)

        # Add label text
        ax.text(
            x_min, y_min - 10, str(label),
            color=label_colors[label],
            fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
        )

    plt.axis('off')
    plt.show(block = False)
    plt.pause(1)

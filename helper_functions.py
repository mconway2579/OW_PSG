import numpy as np
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

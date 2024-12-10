from SceneGraphGeneration import intrinsic_obj, OWLv2, SAM2, get_graph, Node, homog_coord_to_pose_vector, pose_vector_to_homog_coord
from SceneGraphManger import Graph_Manager
from magpie_control.ur5 import UR5_Interface as robot
from magpie_control import realsense_wrapper as real
from openai import OpenAI
from APIKeys import API_KEY
from control_scripts import goto_vec, get_pictures, get_depth_frame_intrinsics
from config import realSenseFPS, topview_vec
import pickle



if __name__ == "__main__":
    owl = OWLv2()
    print(f"{owl=}")

    sam = SAM2()
    print(f"{sam=}")
    
    client = OpenAI(
        api_key= API_KEY,
    )

    myrobot = robot()
    print(f"starting robot from observation")
    myrobot.start()


    myrs = real.RealSense(fps=realSenseFPS)
    myrs.initConnection()

    gm = Graph_Manager()
    inp = "a"
    i = 0
    #goto_vec(myrobot, topview_vec)
    #while inp != "q":
    rgb_img, depth_img = get_pictures(myrs)
    pose = homog_coord_to_pose_vector(myrobot.get_cam_pose())
    depth_scale, K = get_depth_frame_intrinsics(myrs)
    file_name = input("Enter file name: ")
    with open(f"./custon_dataset/{file_name}.pkl", "wb") as file:
        pickle.dump((rgb_img, depth_img, pose, K, depth_scale), file)
    #    graph = get_graph(client, owl, sam, rgb_img, depth_img, pose, K, depth_scale)
    #    gm.add_graph(graph)

    #    inp = input("press q to quit: ")
    #    i+=1
    myrobot.stop()
    myrs.disconnect()

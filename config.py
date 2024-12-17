vit_thresh = 0.001
vit_model_name = "google/owlvit-base-patch32"

gpt_model = "gpt-4o"
gpt_temp = 1

embedding_model = "text-embedding-3-large"
n_state_samples = 1
voxel_size = 0.002 # adjust based on your data

node_match_thresh=0.25
edge_match_thresh=0.25


import math
topview_vec = [-0.16188220333609551, -0.6234229524443915, 0.5474838984217083, 1e-10, -math.pi, -1e-10]
sideview_vec =[0.0360674358115564, -0.20624107287146376, 0.2646274319314355, 1.8434675848139614, 1.4569842711938066, -1.2315497051361715]
righview_vec = [-0.5894860114531234, -0.4276841228860841, 0.37787015136121094, 2.6070220768410595, 0.5874075394228101, 0.8576873740587936]
arm_speed = 0.4
realSenseFPS = 30



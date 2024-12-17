vit_thresh = 0.001 #unused
vit_model_name = "google/owlvit-base-patch32" #vit model to use for bounding boxes

gpt_model = "gpt-4o" #gpt model to use for image understanding
gpt_temp = 1 #gpt tempt parameter


st_embedding_model = 'all-MiniLM-L6-v2' #sentence transformer model to get text embeddings
gpt_embedding_model = "text-embedding-3-large" #gpt model to get text embeddings
n_state_samples = 5 #number of state samples to get from the gpt model in the montecarlo process

voxel_size = 0.002 #voxel size for points and visualisation
 
node_match_thresh=0.25 #Cosin distance between text embeddings for two nodes to be considered the same
edge_match_thresh=0.20 #Cosoin distance between text embeddings for two edges to be considered the same


#Robot stuff
import math
topview_vec = [-0.16188220333609551, -0.6234229524443915, 0.5474838984217083, 1e-10, -math.pi, -1e-10]
sideview_vec =[0.0360674358115564, -0.20624107287146376, 0.2646274319314355, 1.8434675848139614, 1.4569842711938066, -1.2315497051361715]
righview_vec = [-0.5894860114531234, -0.4276841228860841, 0.37787015136121094, 2.6070220768410595, 0.5874075394228101, 0.8576873740587936]
arm_speed = 0.4
realSenseFPS = 30



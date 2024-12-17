import pickle
from helper_functions import display_graph
if __name__ == "__main__":
    G = None
    with open("./PreGeneratedSceneGraphs/PreGeneratedSGM2.pkl", "rb") as file:
        G = pickle.load(file)
    for obj, node in G.nodes(data=True):
        print(f"{node['name']}")
        print(f"{node['data'].points.shape=}")
        print(f"{node['data'].colors.shape=}")
    for u,v, attr in G.edges(data=True):
        print(f"{u=}, {v=}, {attr=}")
    

        
    display_graph(G, blocking=True)
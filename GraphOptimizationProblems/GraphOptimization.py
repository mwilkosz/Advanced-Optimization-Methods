import numpy as np

class DijkstraAlgorithm:
    #Constructor function for prerequisites: Adjacency matrix G, S-initial node, T-target node
    def __init__(self, G, initial_node, target_node):
        self.S = initial_node-1
        self.T = target_node-1
        self.run(G)

    #Updates minimum distance between towns-vertices and stores them in list - dist[].
    #In first iteration of algorithm all 15 distances are assigned as inf. 
    #First minimum distance is also inf, for given starting town distance equals 0 and replaces inf. 
    #In next iterations distances are updated by smaller possible values. 
    #At the end of every function call, returns index of added distance to list.  
    def min_distance(self, dist, vertices): 
        minimum = float("Inf")
        index = None
        for i in range(len(dist)):
            if dist[i] < minimum and i in vertices:
                minimum = dist[i]
                index = i
        return index

    #Prints S-initial node, T-target node and length of path as a Distance. 
    def print_results(self, dist, previous):
        print("Initial node {}\nTarget node: {}\nDistance: {}\nPath:".format(
            self.S+1, self.T+1, dist[self.T]))
        self.print_path(previous, self.T)
    
    #Prints obtained path.
    def print_path(self, previous, j):
        if previous[j] == None:
            print (j+1,)
            return
        self.print_path(previous, previous[j])
        print (j+1,)
    
    #Main algorithm function
    def run(self, G):
        #Copy adjacency matrix G.
        new_G = G
        #Creates dist list of adjacency matrix row length and fills with inf values. 
        dist = np.full(len(new_G), np.inf)
        #List of previously visited vertices
        previous = np.full(len(new_G), None)
        #Assign distance to initial node. 
        dist[self.S] = 0
        #Create towns - graph vertices. 
        vertices = np.arange(0,len(new_G))
        #Executing until size of processed vertices list is greater than 0. 
        while vertices.size > 0:
            #Takes index of minimal distance between towns in distance list. 
            u = self.min_distance(dist,vertices)
            #Delete last downloaded vertex index from vertices. 
            vertices = np.delete(vertices, np.argwhere(vertices == u))
            #Iteration over town - vertices. 
            for i in range(len(new_G)):
                #If there exist an edge between element of adjacency matrix which is also in list of vertices-towns.
                if new_G[u][i] and i in vertices:
                    #If distance of index u + disance between neighbour and considered node is smaller then distance in dist list. 
                    if dist[u] + new_G[u][i]< dist[i]:
                        #Element i of dist list is now dist[u] with distance between considered node and neighbour.
                        dist[i] = dist[u] + new_G[u][i]
                        #List previous[] is updated with added index.
                        previous[i] = u

        self.print_results(dist, previous)

if __name__ == "__main__":
    
    #Adjacency matrix
    G = np.array(
        [[  0.,  78.,   0.,   0.,   0., 203.,   0.,   0.,   0.,   0.,   0., 0.,   0.,   0., 113.],
        [ 78.,   0., 168., 294.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 0.,   0.,   0.,   0.],
        [  0., 168.,   0., 332.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 0.,   0.,   0.,   0.],
        [  0., 294., 332.,   0., 214., 130., 260.,   0.,   0.,   0.,   0., 0.,   0.,   0.,   0.],
        [  0.,   0.,   0., 214.,   0.,   0., 170.,   0., 167.,   0.,   0., 0.,   0.,   0.,   0.],
        [203.,   0.,   0., 130.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 218.,   0.,   0.,   0.],
        [  0.,   0.,   0., 260., 170.,   0.,   0.,  46., 170.,   0.,   0., 0.,   0.,   0.,   0.],
        [  0.,   0.,   0.,   0.,   0.,   0.,  46.,   0., 175.,   0., 215., 139.,   0.,   0.,   0.],
        [  0.,   0.,   0.,   0., 167.,   0., 170., 175.,   0., 356.,   0., 0.,   0.,   0.,   0.],
        [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 356.,   0., 105., 0.,   0.,   0.,   0.],
        [  0.,   0.,   0.,   0.,   0.,   0.,   0., 215.,   0., 105.,   0., 0., 112.,   0.,   0.],
        [  0.,   0.,   0.,   0.,   0., 218.,   0., 139.,   0.,   0.,   0., 0.,   0., 182., 285.],
        [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 112., 0.,   0., 187.,   0.],
        [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 182., 187.,   0.,  98.,   0.],
        [113.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 285.,   0.,  98.,   0.]])

    dij = DijkstraAlgorithm(G, initial_node=2, target_node=9)
    input("Press Enter to continue...")
    
    pass
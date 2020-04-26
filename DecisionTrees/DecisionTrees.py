import numpy as np

class ZeroSumGames():
    
    def __init__(self):
        self.final_matrix=None
        self.matrix=None
        self.n=0
        self.saddle_points_list=[]

    def saddle_point(self,matrix):
        #Finds saddle point of given matrix.
        D1=matrix.sum(axis=1).argmax()
        #D1 maximizes. Returns row coordinate of the maxmin element.
        D2=matrix.sum(axis=0).argmin()
        #D2 minimizes. Returns column coordinate of the minmax element.
        matrix_saddle_point=matrix[D1,D2]
        return matrix_saddle_point

    def create_arrays(self,n):
        #Creates n-random matrix of size 2x2 and calculate their saddle points.
        #The result of this function is a list of saddle points.
        for i in range(n):
            A=np.random.randint(100,size=(2,2))
            print(i+1,"randomly generated matrix of size 2x2 is {}".format(A))
            self.saddle_points_list.append(self.saddle_point(A))
        print("Saddle points of generated matrices are {}".format(self.saddle_points_list))
        return self.saddle_points_list

    def result(self):
        self.create_arrays(6)
        #Creates 6 random matrices according to the last(5th) tree structure.
        self.final_matrix=np.empty((2,3))
        #Creates final matrix of dimension obtained for given tree.
        with np.nditer(self.final_matrix, op_flags=["readwrite"]) as it:
            pos=0
            for x in it:
                x[...]=self.saddle_points_list[pos]
                pos+=1
        #Insert saddle points into the final matrix.
        print("\nThe final matrix formed from the saddle points of each matrix {}".format(self.final_matrix))
        print("\nThe result of considered decision tree is {}".format(self.saddle_point(self.final_matrix)))

if __name__ == "__main__":
    decision_tree=ZeroSumGames()
    decision_tree.result()

    pass

#SOLUTION OF RANDOMLY GENERATED EXAMPLE#

# 1 randomly generated matrix of size 2x2 is [[69  8]
#                                             [89 38]]
# 2 randomly generated matrix of size 2x2 is [[18 97]
#                                             [98 21]]
# 3 randomly generated matrix of size 2x2 is [[16 94]
#                                             [67 33]]
# 4 randomly generated matrix of size 2x2 is [[20 23]
#                                             [91 25]]
# 5 randomly generated matrix of size 2x2 is [[34 48]
#                                             [30 38]]
# 6 randomly generated matrix of size 2x2 is [[45 65]
#                                             [85 99]]
# Saddle points of generated matrices are [38, 98, 16, 25, 34, 85]

# The final matrix formed from the saddle points of each matrix [[38. 98. 16.]
#                                                                [25. 34. 85.]]

# The result of considered decision tree is 38.0
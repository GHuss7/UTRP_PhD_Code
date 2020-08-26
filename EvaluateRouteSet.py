#Evaluates route sets and plots instance and routes
#Ror the Urban Transit Routing Problem
#Author: Christine Mumford 
#email: MumfordCL@cardiff.ac.uk
#Date: 14th October 2017

#Input 3 Text files:

#************FIRST FILE
#A matrix of travel times between directly connected nodes (bus stops)
#infinity, 'inf' denotes no direct connection

#************SECOND FILE
#A matrix of passenger demand between all pairs of nodes (bus stops)
#Denoting number of travellers per unit time between each source and destination

#************THIRD FILE
#Route Set File

#EITHER for readfiles
#Each route on a new line
#Consisting of integer labels
#Either between 0 and n, or 0 and (n-1)
#Where n = number of nodes (bus stops)

#OR for readfilesALT
#Route sets are in python list of lists

#Assumes that travel time and demand matrices are symmetrical

#Transfer penalty = 5 mins by default
#Waiting time = 0 

#Written in Python 3
#Requires numpy and tkinter
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from numpy import array,loadtxt,isfinite,set_printoptions,zeros,ones,int,empty,inf,asarray,minimum,newaxis
import numpy as np


def main():

    '''Read files'''
    travelTimes, DemandMat, routes,smallest = readfiles()
    n = n = len(DemandMat)
    print('n =',n)
    total_demand = sum(sum(DemandMat))/2
    '''Get user input'''
    tp = getusrinput()
    
    '''Convert route sets to start at node 0 '''
    routes, smallest = standardize(routes)
    
    '''Check validity of route set'''
    connected, allPresent, duplicates = checkValidity(routes,n)
    print('validity checked')
    if (not connected) or (not allPresent):
        print('invalid route set')
        if not connected:
            print('not connected')
        if not allPresent:
            print('one or more nodes is missing')
    
        return()
    if duplicates:
            print('one or more node is duplicated in a single route making a cycle')
    '''waiting time, wt, is different from transfer penalty,tp'''
    wt = 0 #waiting time, can be changed if desired
    ATT,d0,d1,d2,drest,noRoutes,longest,shortest = fullPassengerEvaluation(routes, \
                        travelTimes,DemandMat, total_demand,n,tp,wt)
    RL = evaluateTotalRouteLength(routes,travelTimes)
    if RL > 0:
        print('ATT =',ATT)
        print('d0 =',d0)
        print('d1 =',d1)
        print('d2 =',d2)
        print('dun =',drest)
        print('RL =',RL)
        print('No of routes = ',noRoutes)
        print('Shortest route is',shortest,'nodes long')
        print('Longest route is',longest,'nodes long')
    else:
        print('illegal edge present...')
    input('Press any key to continue ')
    
def getusrinput():
    root = tk.Tk()
    root.update()
    root.withdraw()
    tp = simpledialog.askinteger(
        "Waitng time/transfer penalty", "Enter the tranfer penalty in minutes", 
    initialvalue=5)
    root.destroy()
    return(tp)

def linkfile(TypeOfFile):
    root = tk.Tk()
    root.update()
    root.withdraw()
    messagebox.showinfo(message="Select a file holding the %s" % TypeOfFile)
    filename = filedialog.askopenfilename()
    root.destroy()
    return filename

def readfiles():
    filename = linkfile('Travel Times')
    travelTimesFile = open(filename,'r')
    travelTimes = loadtxt(travelTimesFile)
    travelTimesFile.close()
    filename = linkfile('Passenger Demand')
    DemandFile = open(filename,'r')
    DemandMat = loadtxt(DemandFile)
    DemandFile.close()
    filename = linkfile('Route set for evaluation')
    with open(filename) as f:
        routes = []
        for line in f:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [int(i) for i in line]
                routes.append(line)
    routes,smallest = standardize(routes)
    return(travelTimes,DemandMat,routes,smallest)
    
def readfilesALT():
    filename = linkfile('Travel Times')
    travelTimesFile = open(filename,'r')
    travelTimes = loadtxt(travelTimesFile)
    travelTimesFile.close()
    filename = linkfile('Passenger Demand')
    DemandFile = open(filename,'r')
    DemandMat = loadtxt(DemandFile)
    DemandFile.close()
    filename = linkfile('Route set for evaluation')
    f = open(filename, 'r')
    routes = eval(f.read())

    routes,smallest = standardize(routes)
    return(travelTimes,DemandMat,routes,smallest)
    
def standardize(routes):
    '''numbering convention may start from node 0 or node 1'''
    '''convert all to start at route 0'''
    largest = 0
    smallest = float('Inf')
    r = len(routes)
    for i in range(0,r):
        if max(routes[i]) > largest:
            largest = max(routes[i])
        if min(routes[i]) < smallest:
            smallest = min(routes[i])
    if smallest > 0:
        for i in range(0,r):
            l = len(routes[i])
            for j in range(0,l):
                routes[i][j] = routes[i][j]-smallest
    return(routes,smallest)
        
    
def checkValidity(routes,n):
    connected,allPresent = isconnected(routes,n) # Check connectivity and all nodes present
    print('Connected is',connected)
    print('AllPresent is',allPresent)
    duplicates,i = checkDuplicates(routes)
    print('duplicates is',duplicates)
    if duplicates:
        print('duplicate node in route: ',i)
    return(connected,allPresent,duplicates)
    
def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

    
def MakeList(n,EdgeWeight):
    adj_list = [[] for i in range(n)]
    for i in range(n-1):
        for j in range(i+1,n):
            if isfinite(EdgeWeight[i,j]):
                adj_list[i].append(j)
                adj_list[j].append(i)
    return(adj_list)
    
def floyd_warshall_fastest(mat,t):
    
    #mat, t = check_and_convert_adjacency_matrix(m)
    for k in range(t):
        mat = minimum(mat, mat[newaxis,k,:] + mat[:,k,newaxis]) 

    return mat
    
def EvaluateATT(SPMatrix,Demand, total_demand, WT):
    
    total_ATT = total_demand*WT
    total_travel = Demand*SPMatrix/2
    total_ATT = total_ATT + sum(sum(total_travel))
    ATT = total_ATT/total_demand
    return(ATT)
    
    
def convertToDict(adj,TT,n):
    temp_adj = list(adj)
    for i in range(len(adj)): # remove duplicate edges, assuming undirected graph
        temp_adj[i] = [s for s in adj[i] if s > i]
    #Convert ajacency list to a dictionary
    G = {x:{temp_adj[x][i]:TT[x][temp_adj[x][i]] for i in range(len(temp_adj[x]))} for x in range(n)}
    return(G)
    
    
def MinimumSpanningTree(G):
    """
    Return the minimum spanning tree of an undirected graph G.
    G should be represented in such a way that G[u][v] gives the
    length of edge u,v, and G[u][v] should always equal G[v][u].
    The tree is returned as a list of edges.
    """
 
    # Kruskal's algorithm: sort edges by weight, and add them one at a time.
    # We use Kruskal's algorithm, first because it is very simple to
    # implement once UnionFind exists, and second, because the only slow
    # part (the sort) is sped up by being built into Python.
    subtrees = UnionFind()
    tree = []
    edges = [(G[u][v],u,v) for u in G for v in G[u]]
    edges.sort()
    length = 0
    for W,u,v in edges:
        if subtrees[u] != subtrees[v]:
            tree.append((u,v))
            subtrees.union(u,v)
            length = length + W
    return tree,length,edges
    


                
def expandTravelMatrix(routes, travelTimes,n,tp,wt):
    set_printoptions(threshold= 200)
    t = int(0); # t give the sum of the number of nodes in all the routes
    r = len(routes) # Number of routes
    routelength = zeros((r,), dtype=int)
    shortest = inf
    longest = 0
    for i in range(r):
        length = len(routes[i])
        routelength[i] = length
        t = t + length
        if length < shortest:
            shortest = length
        if length > longest:
            longest = length
    #print('routelength array',routelength)
    routeadj = empty((t,t))
    routeadj[:] = inf
    displacement = 0
    mapping = [[]]*n
    inv_map = zeros((t,), dtype=int)
    for i in range(r):
        for j in range(routelength[i]-1):
            p1 = routes[i][j]
            p2 = routes[i][j+1]
            q1 = j + displacement
            q2 = j + 1 + displacement
            routeadj[q1][q2] = travelTimes[p1][p2]
            routeadj[q2][q1] = travelTimes[p2][p1]
            mapping[p1]= mapping[p1] + [q1]
            mapping[p2]= mapping[p2] + [q2]
            inv_map[q1] = p1
            inv_map[q2] = p2
        displacement = displacement + routelength[i]
    # add the 5 minute delays for vehicle changes
    for i in range(n):
        for j in range(int(len(mapping[i]))-1):
            for k in range((j+1),int(len(mapping[i]))):
                q1 = mapping[i][j]
                q2 = mapping[i][k]
                routeadj[q1][q2] = tp + wt
                routeadj[q2][q1] = tp + wt
                
    routeadj = array(routeadj)
    return(routeadj,inv_map,t,shortest,longest)  
    
def shortest_paths_matrix(D, inv_map, t, n):

    SPMatrix = inf*ones((n,n), dtype=float)
    #count = 0
    for i in range(t):
        p1 = inv_map[i]
        for j in range(t):
            p2 = inv_map[j]
            if (D[i][j]<SPMatrix[p1][p2]):
                SPMatrix[p1][p2] = D[i][j]
                #count = count + 1
    return(SPMatrix)
    
def evaluateTotalRouteLength(routes,travelTimes):
    #calc_fitnessRL sums all the route lengths
   
    RL=0
    r = len(routes)
    routelength = zeros((r,), dtype=int)
    
    for i in range(r):
        routelength[i] = len(routes[i])
    for i in range(r):
        for j in range(routelength[i]-1):
            new_edge = travelTimes[routes[i][j],routes[i][j+1]]
            if isfinite(new_edge):
                RL = RL + new_edge
            else:
                print('Invalid route set')
                break
    return(RL)
    
    
def evaluateObjectives(routeset,travelTimes,DemandMat,total_demand,n,r,wt,tp):
    RL = evaluateTotalRouteLength(routeset,travelTimes)
    routeadj,inv_map,t,shortest,longest = expandTravelMatrix(routeset, travelTimes,n,tp,wt)
    D = floyd_warshall_fastest(routeadj,t)
    SPMatrix = shortest_paths_matrix(D, inv_map, t, n)
    ATT = EvaluateATT(SPMatrix, DemandMat, total_demand, wt)
    return(ATT,RL)
    
    
def evalObjs(routeset,travelTimes,DemandMat,parameters_input):
    total_demand = parameters_input['total_demand']
    n = parameters_input['n'] # number of nodes
    wt = parameters_input['wt'] # waiting time
    tp = parameters_input['tp'] # transfer penalty
    
    RL = evaluateTotalRouteLength(routeset,travelTimes)
    routeadj,inv_map,t,shortest,longest = expandTravelMatrix(routeset, travelTimes,n,tp,wt)
    D = floyd_warshall_fastest(routeadj,t)
    SPMatrix = shortest_paths_matrix(D, inv_map, t, n)
    ATT = EvaluateATT(SPMatrix, DemandMat, total_demand, wt)
    return ATT, RL
        
def isconnected(routes,n):   
    ''''Routine to check whether candidate route set is connected'''
    ''' assumes numbering of nodes starts at zero'''
    temproutes = list(routes)
    connected_nodes = set(temproutes[0]) # registers nodes connected in growing route network
    temproutes.pop(0)
    connected = True
    while temproutes != [] and connected:
        connected = False
        processedRoutes = list(temproutes)
        for route in temproutes:
            if connected_nodes & set(route) != set():
                connected_nodes = connected_nodes | set(route)
                processedRoutes.remove(route)
                connected = True
            temproutes = list(processedRoutes)
        if len(connected_nodes) == n:
            allPresent = True
        else:
            allPresent = False
    if not allPresent:
        connected_nodes = {x + 1 for x in connected_nodes}
        all = set(range(1,(n+1)))
        missing_nodes = all - connected_nodes
        print('missing nodes \n',missing_nodes)
    return(connected,allPresent)
    
def checkDuplicates(routes):
    duplicates = False
    i = 1
    for route in routes:
        if len(route) != len(set(route)):
            duplicates = True
            break
        i = i + 1
    return(duplicates,i)
        
    
def writeRoutesFile(Routeset,name):

    with open(name,'w') as f:
        for route in Routeset:
            f.write(' '.join(map(str, route)) + '\n')
            
            
def expandTravelMatrixChanges(routes, travelTimes,n,tp,wt):
    set_printoptions(threshold= 200)
    t = int(0); # t give the sum of the number of nodes in all the routes
    r = len(routes) # Number of routes
    routelength = zeros((r,), dtype=int)
    shortest = inf
    longest = 0
    for i in range(r):
        length = len(routes[i])
        routelength[i] = length
        t = t + length
        if length < shortest:
            shortest = length
        if length > longest:
            longest = length
    #print('routelength array',routelength)
    routeadj = empty((t,t))
    routeadj[:] = inf
    changes = zeros((t,t),dtype=int)
    displacement = 0    # displacement keeps track of where in the adj matrix you are busy
    mapping = [[]]*n    # mapping shows where all the corresponding nodes are found on the route adj matrix
    inv_map = zeros((t,), dtype=int)
    for i in range(r):
        for j in range(routelength[i]-1):
            p1 = routes[i][j]   # gets the first node index of the examined edge
            p2 = routes[i][j+1] # gets the second node index of the examined edge
            q1 = j + displacement       # sets the one index for the expanded route adj matrix
            q2 = j + 1 + displacement   # sets the other index for the expanded route adj matrix
            routeadj[q1][q2] = travelTimes[p1][p2]  # populate the route adj matrix both ways
            routeadj[q2][q1] = travelTimes[p2][p1]  # populate the route adj matrix both ways
            mapping[p1]= mapping[p1] + [q1]         # adds the routes adj matrix index to the correct mapping position of the node involved
            mapping[p2]= mapping[p2] + [q2]         # adds the routes adj matrix index to the correct mapping position of the node involved
            inv_map[q1] = p1                        # adds the inverse of the map to keep track of what node is represented in the route adj matrix
            inv_map[q2] = p2                        # adds the inverse of the map to keep track of what node is represented in the route adj matrix
        displacement = displacement + routelength[i] # increment the displacement according
                                                        # to the current route's length
    # add the 5 minute delays for vehicle changes
    
    # note to self: the mapping contains doubles and leads to redundant additions, but is probably because of
    # the node i to i transfers and it's a cost to pay 
    
    # mapping contains all of the same nodes but spread over the different routes in the adj matrix
    
    for i in range(n):
        for j in range(int(len(mapping[i]))-1):
            for k in range((j+1),int(len(mapping[i]))):
                q1 = mapping[i][j]                  # not sure about how mapping works ???
                q2 = mapping[i][k]                  # not sure about how mapping works ???
                routeadj[q1][q2] = tp + wt          # adds the penalties to the adj matrix where transfers would occur
                routeadj[q2][q1] = tp + wt          # adds the penalties to the adj matrix where transfers would occur
                changes[q1][q2] = 1                 # indicates where transfers would occur
                changes[q2][q1] = 1                 # indicates where transfers would occur

    routeadj = array(routeadj)
    changes = array(changes)
    return(routeadj,changes,inv_map,t,shortest,longest)
                
def check_and_convert_adjacency_matrix(adjacency_matrix,changes):
    mat = asarray(adjacency_matrix)

    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows
    #fill_diagonal(mat,0.0)
    #assert (diagonal(mat) == 0.0).all()
    mat = array(mat)
    changes = array(changes)
    return (mat, changes, n)
                
def FastFloydChanges(m,changes):
    m, changes,t = check_and_convert_adjacency_matrix(m,changes)
    for k in range(t):
        new_m = m[newaxis,k,:] + m[:,k,newaxis]             # adds the two vectors to each other over the nxn matrix
        new_c = changes[newaxis,k,:] + changes[:,k,newaxis] # adds the two vectors to each other over the nxn matrix
        bool1 = (m > new_m) + zeros((t,t),dtype=int)        # boolean for where the adj mx is > than the new m mx
        bool2 = (m<= new_m) + zeros((t,t),dtype=int)        # boolean for where the adj mx is <= than the new m mx
        bool3 = (m == new_m) + zeros((t,t),dtype=int)       # boolean for where the adj mx is = to the new m mx
        bool4 = (changes > new_c) + zeros((t,t),dtype=int)  # boolean for where the changes mx is > than the new c mx
        bool5 = bool3*bool4
        bool1 = bool1 + bool5
        bool2 = bool2 - bool5
        temp1 = bool1*new_c
        temp2 = bool2*changes
        changes = temp1 + temp2
        m = minimum(m, new_m)
    return(m, changes)

def shortest_paths_matrixChanges(D, inv_map, t, n, changes):

    SPMatrix = inf*ones((n,n), dtype=float)
    ChMatrix = zeros((n,n), dtype=int)
    #count = 0
    for i in range(t):
        p1 = inv_map[i]
        for j in range(t):
            p2 = inv_map[j]
            if (D[i][j]<SPMatrix[p1][p2]) or ((D[i][j]==SPMatrix[p1][p2]) and (changes[i][j]<ChMatrix[p1][p2])):
                SPMatrix[p1][p2] = D[i][j]
                ChMatrix[p1][p2] = changes[i][j]
                #count = count + 1
    return(SPMatrix,ChMatrix)
    
def EvaluateChanges(n, SPMatrix, ChMatrix, Demand, total_demand, TP, WT):
    #Evaluate: Evaluates the details of the paraments, d_0, d_1, d_2, d_rest,
    #ATT total in vehicle travel time, total transfer time, ATT etc.
    
    total_travel = Demand*SPMatrix/2
    bool = (ChMatrix == zeros((n,n), dtype=int))+zeros((n,n), dtype=int)
    total_d0 = Demand*bool;
    d0 = sum(sum(total_d0))/2;
    bool = (ChMatrix == ones((n,n),dtype=int))+zeros((n,n), dtype=int)
    total_d1 = Demand*bool;
    d1 = sum(sum(total_d1))/2;
    bool = (ChMatrix == 2*ones((n,n),dtype=int))+zeros((n,n), dtype=int)
    total_d2 = Demand*bool
    d2 = sum(sum(total_d2))/2;
    bool = (ChMatrix > 2*ones((n,n),dtype=int))+zeros((n,n), dtype=int)
    total_drest = Demand*bool;
    drest = sum(sum(total_drest))/2
    total_ATT = sum(sum(total_travel));
    ATT = total_ATT/total_demand
    d0 = d0*100/total_demand
    d1 = d1*100/total_demand
    d2 = d2*100/total_demand
    drest = drest*100/total_demand
    Temp = Demand*ChMatrix/2
    total_TT = TP*sum(sum(Temp))
    total_ATT = 2*total_ATT
    total_TT = 2*total_TT
    return(d0, d1, d2, drest, ATT, total_ATT, total_TT)
    
def fullPassengerEvaluation(routes, travelTimes,DemandMat, total_demand,n,tp,wt):
    routeadj,changes,inv_map,t,shortest,longest = expandTravelMatrixChanges(routes, travelTimes,n,tp,wt)
    D, changes = FastFloydChanges(routeadj,changes)
    SPMatrix,ChMatrix = shortest_paths_matrixChanges(D, inv_map, t, n, changes)
    #print(ChMatrix)
    a = np.array(ChMatrix)
    np.savetxt('ChangesMat.txt', a.astype(int), fmt='%i', delimiter=' ')   # X is an array
    d0, d1, d2, drest, ATT, total_ATT, total_TT = EvaluateChanges(n, SPMatrix, ChMatrix, DemandMat, total_demand, tp, wt)
    noRoutes = len(routes)
    return(ATT,d0,d1,d2,drest,noRoutes,longest,shortest)
    
                   
# main()

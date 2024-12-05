import numpy as np

# The goal of this routine is to return the minimum cost dynamic programming
# solution given a set of unary and pairwise costs

def dynamicProgram(unaryCosts, pairwiseCosts):

    # Count number of positions  (i.e. pixels in the scanline), 
    # and nodes at each position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = unaryCosts.shape[0] # Number of rows = number of possible disparities
    nPosition = unaryCosts.shape[1] # Number of columns = number of positions

    # Define minimum cost matrix 
    # each element will eventually contain the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCostMatrix = np.zeros([nNodesPerPosition, nPosition])

    # Define parent matrix 
    # each element will contain the (vertical) index of the node that preceded it on the path.  
    # Since the first column has no parents, we will leave it set to zeros.
    parents = np.zeros([nNodesPerPosition, nPosition])

    ###################### FORWARD PASS ######################
    # First column of minimumCostMatrix is just the unary costs
    # i.e. there is no pairwise cost for the first column
    minimum = np.zeros([nNodesPerPosition,1]) 
    for i in range(nNodesPerPosition):
        minimum[i,0] = unaryCosts[i][0] # Assign the unary cost of the node to the minimum array 
    minimumCostMatrix[:,0] = minimum[:,0]

    # Iterate through the minimumCostMatrix to find the minimum cost to reach each node
    for j in range(1,nPosition): # Columns of the minimumCostMatrix: starting from the second column
        for i in range(nNodesPerPosition): # Rows of the minimumCostMatrix

            # Calculate the possible path costs from each possible previous node to the current node
            # possPathCosts =  Minimum cost to reach previous node 
            #                  + Unary cost of current node 
            #                  + pairwise cost from previous node to current node
            possPathCosts = np.zeros([nNodesPerPosition,1]) 
            for n in range(nNodesPerPosition):
                possPathCosts[n,0] = minimumCostMatrix[n,j-1] + unaryCosts[i][j] + pairwiseCosts[n][i]
                
            # Update the minimumCostMatrix with the minimum cost and the parent index
            minimumCostMatrix[i,j] = np.min(possPathCosts)
            parents[i,j] = np.argmin(possPathCosts)

    ###################### BACKWARD PASS ######################
    # Find the best path by tracing back through the minimumCostMatrix
    bestPath = np.zeros([nPosition,1]) # Initialize the best path array
    
    # The minimum cost to reach the last column is the minimum of the last column of the minimumCostMatrix
    lastColumn = minimumCostMatrix[:,-1]
    minCost = np.min(lastColumn) # Minimum cost in the last column
    minInd = np.argmin(lastColumn) # Index of the minimum cost in the last column
    
    # The best path to each the last column and its parent index
    bestPath[-1] = minInd # Best path in the last column
    bestParentInd = int(parents[minInd,-1]) # Parent index of the best path in the last column
   
    # Run backwards through the cost matrix tracing the best path
    for j in range(nPosition-2,-1,-1):
        # Assign the best path to the current node
        bestPath[j] = bestParentInd 
        # Get the best parent index of the current node : for next iteration
        bestParentInd = int(parents[bestParentInd,j]) # Update the best parent index according to the parent matrix

    return bestPath


def dynamicProgramVec(unaryCosts, pairwiseCosts):
    # Count number of positions and nodes per position
    nNodesPerPosition = unaryCosts.shape[0]
    nPosition = unaryCosts.shape[1]

    # Initialize minimum cost matrix and parent matrix
    minimumCostMatrix = np.zeros([nNodesPerPosition, nPosition])
    parents = np.zeros([nNodesPerPosition, nPosition], dtype=int)

    ###################### FORWARD PASS ######################
    # First column of minimumCostMatrix is just the unary costs
    minimumCostMatrix[:, 0] = unaryCosts[:, 0]

    # Compute minimum cost column by column
    for j in range(1, nPosition):
        # Add the unary costs of the current column to the minimum costs of the previous column
        unaryColumn = unaryCosts[:, j].reshape(1, -1)  # Shape (1, nNodesPerPosition)

        # Add the pairwise costs for every possible transition
        previousCosts = minimumCostMatrix[:, j-1].reshape(-1, 1)  # Shape (nNodesPerPosition, 1)
        transitionCosts = previousCosts + pairwiseCosts  # Shape (nNodesPerPosition, nNodesPerPosition)

        # Add the unary costs to the transition costs
        totalCosts = transitionCosts + unaryColumn  # Shape (nNodesPerPosition, nNodesPerPosition)

        # Find the minimum cost for each node in this column
        minimumCostMatrix[:, j] = np.min(totalCosts, axis=0)
        parents[:, j] = np.argmin(totalCosts, axis=0)

    ###################### BACKWARD PASS ######################
    # Trace the best path by backtracking through the parent matrix
    bestPath = np.zeros(nPosition, dtype=int)

    # Start with the node in the last column with the smallest cost
    bestPath[-1] = np.argmin(minimumCostMatrix[:, -1])

    # Trace back through the parent matrix
    for j in range(nPosition - 2, -1, -1):
        bestPath[j] = parents[bestPath[j+1], j+1]

    return bestPath
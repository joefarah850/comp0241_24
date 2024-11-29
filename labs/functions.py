import numpy as np
# the goal of this routine is to return the minimum cost dynamic programming
# solution given a set of unary and pairwise costs
def dynamicProgram(unaryCosts, pairwiseCosts):

    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])

    # define parent matrix - each element will contain the (vertical) index of
    # the node that preceded it on the path.  Since the first column has no
    # parents, we will leave it set to zeros.
    parents = np.zeros([nNodesPerPosition, nPosition], dtype=int)

    # FORWARD PASS

    # TODO:  fill in first column of minimum cost matrix
    minimumCost[:, 0] = unaryCosts[:, 0]

    # Now run through each position (column)
    for cPosition in range(1,nPosition):
        # run through each node (element of column)
        for cNode in range(nNodesPerPosition):
            # now we find the costs of all paths from the previous column to this node
            possPathCosts = np.zeros([nNodesPerPosition, 1])
            for cPrevNode in range(nNodesPerPosition):
                # TODO  - fill in elements of possPathCosts
                possPathCosts[cPrevNode,0] = (minimumCost[cPrevNode, cPosition - 1] + pairwiseCosts[cPrevNode, cNode] + unaryCosts[cNode, cPosition])

            # TODO - find the minimum of the possible paths 
            minCost = np.min(possPathCosts)
            ind = np.argmin(possPathCosts)
            
            # Assertion to check that there is only one minimum cost.
            # assert(len(np.where(possPathCosts == minCost)[0]) == 1)

            # TODO - store the minimum cost in the minimumCost matrix
            minimumCost[cNode, cPosition] = minCost 
            
            # TODO - store the parent index in the parents matrix
            parents[cNode, cPosition] = ind

    #BACKWARD PASS

    #we will now fill in the bestPath vector
    bestPath = np.zeros([nPosition,1])
    
    #TODO  - find the index of the overall minimum cost from the last column and put this
    #into the last entry of best path
    minCost = np.min(minimumCost[:, -1])
    minInd = np.argmin(minimumCost[:, -1])
    bestPath[-1] = minInd

    # TODO - find the parent of the node you just found
    bestParent = parents[minInd, -1]

    # run backwards through the cost matrix tracing the best patch
    for cPosition in range(nPosition-2,-1,-1):
        # TODO - work through matrix backwards, updating bestPath by tracing parents
        bestPath[cPosition] = bestParent 
        bestParent = int(parents[bestParent, cPosition])

    return bestPath

def dynamicProgramVec(unaryCosts, pairwiseCosts):
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # Define minimum cost matrix
    minimumCost = np.zeros([nNodesPerPosition, nPosition])

    # Define parent matrix
    parents = np.zeros([nNodesPerPosition, nPosition], dtype=int)

    # Fill the first column of minimumCost
    minimumCost[:, 0] = unaryCosts[:, 0]

    # FORWARD PASS
    for cPosition in range(1, nPosition):
        # Tile previous costs
        # Tile Repeats these costs for all current nodes, creating a matrix of shape (nNodesPerPosition, nNodesPerPosition) where:
        #       •	Rows correspond to the previous nodes.
        #       •	Columns correspond to the current nodes.
        prevCosts = np.tile(minimumCost[:, cPosition - 1], (nNodesPerPosition, 1)).T

        # Compute total costs
        totalCosts = prevCosts + pairwiseCosts + unaryCosts[:, cPosition][None, :]

        # Find minimum costs and parent indices
        minCosts = np.min(totalCosts, axis=0)
        parentIndices = np.argmin(totalCosts, axis=0)

        # Update minimumCost and parents
        minimumCost[:, cPosition] = minCosts
        parents[:, cPosition] = parentIndices

    # BACKWARD PASS
    bestPath = np.zeros([nPosition, 1], dtype=int)

    # Find the starting point
    minInd = np.argmin(minimumCost[:, -1])
    bestPath[-1] = minInd

    # Trace back the path
    for cPosition in range(nPosition - 2, -1, -1):
        bestPath[cPosition] = parents[bestPath[cPosition + 1], cPosition + 1]

    return bestPath
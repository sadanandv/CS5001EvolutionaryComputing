import preq
from typing import List, Tuple, Dict
from collections import deque

def hill_climbing(graph: Dict[str, List[str]], start: str, goal: str, heuristic: Dict[str, int]) -> List[str]:
    """
    Hill Climbing Algorithm to find a path from start to goal in a graph based on heuristic values.
    
    Parameters:
    - graph: A dictionary representing the graph. Keys are node names, and values are lists of neighbors.
    - start: The start node.
    - goal: The goal node.
    - heuristic: A dictionary representing the heuristic values for each node.
    
    Returns:
    - A list representing a path from start to goal based on the heuristic values.
    """
    
    # Initialize the current node and path
    current_node = start
    path = [current_node]
    
    while current_node != goal:
        # Get the neighbors of the current node
        neighbors = graph.get(current_node, [])
        
        # If there are no neighbors, we're stuck
        if not neighbors:
            return None
        
        # Find the neighbor with the lowest heuristic value
        best_neighbor = min(neighbors, key=lambda x: heuristic.get(x, float('inf')))
        
        # If the best neighbor has a heuristic value greater than the current node, we're stuck
        if heuristic.get(best_neighbor, float('inf')) >= heuristic.get(current_node, float('inf')):
            return None
        
        # Move to the best neighbor
        current_node = best_neighbor
        path.append(current_node)
        
    return path

# Example usage

result = hill_climbing(preq.example_graph, preq.start_node, preq.goal_node, preq.example_heuristic)
print(result)



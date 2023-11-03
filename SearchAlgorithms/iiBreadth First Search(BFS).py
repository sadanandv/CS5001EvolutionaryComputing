from collections import deque
from typing import List, Tuple, Dict

def breadth_first_search(graph: Dict[str, List[str]], start: str, goal: str) -> List[str]:
    """
    Breadth First Search Algorithm to find the shortest path from start to goal in a graph.
    
    Parameters:
    - graph: A dictionary representing the graph. Keys are node names, and values are lists of neighbors.
    - start: The start node.
    - goal: The goal node.
    
    Returns:
    - A list representing the shortest path from start to goal.
    """
    
    # Initialize a queue to keep track of the paths to be explored
    queue = deque([[start]])
    
    # Loop until the queue is empty
    while queue:
        current_path = queue.popleft()
        current_node = current_path[-1]
        
        # If the goal is reached, return the path
        if current_node == goal:
            return current_path
        
        # Add the neighbors to the queue for exploration
        for neighbor in graph.get(current_node, []):
            if neighbor not in current_path:
                new_path = current_path + [neighbor]
                queue.append(new_path)
                
    return None


# Example usage
example_graph = {
    'A': ['B', 'C', 'E'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F', 'G'],
    'D': ['B'],
    'E': ['A', 'B', 'D'],
    'F': ['C'],
    'G': ['C']
}
start_node = 'A'
goal_node = 'G'
result = breadth_first_search(example_graph, start_node, goal_node)
print(result)
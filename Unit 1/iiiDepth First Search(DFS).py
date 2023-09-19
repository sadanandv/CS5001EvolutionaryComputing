from typing import List, Tuple, Dict

def depth_first_search(graph: Dict[str, List[str]], start: str, goal: str) -> List[str]:
    """
    Depth First Search Algorithm to find a path from start to goal in a graph.
    
    Parameters:
    - graph: A dictionary representing the graph. Keys are node names, and values are lists of neighbors.
    - start: The start node.
    - goal: The goal node.
    
    Returns:
    - A list representing a path from start to goal.
    """
    
    # Initialize a stack to keep track of the path being explored
    stack = [[start]]
    
    # Loop until the stack is empty
    while stack:
        current_path = stack.pop()
        current_node = current_path[-1]
        
        # If the goal is reached, return the path
        if current_node == goal:
            return current_path
        
        # Add the neighbors to the stack for exploration
        for neighbor in graph.get(current_node, []):
            if neighbor not in current_path:
                new_path = current_path + [neighbor]
                stack.append(new_path)
                
    return None

# Example usage

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
result = depth_first_search(example_graph, start_node, goal_node)
print(result)



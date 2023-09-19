import preq
from typing import List, Tuple, Dict



def british_museum_search(graph: Dict[str, List[str]], start: str, goal: str) -> List[str]:
    """
    British Museum Search Algorithm to find the shortest path from start to goal in a graph.
    
    Parameters:
    - graph: A dictionary representing the graph. Keys are node names, and values are lists of neighbors.
    - start: The start node.
    - goal: The goal node.
    
    Returns:
    - A list representing the shortest path from start to goal.
    """
    
    # Initialize a list to keep track of all possible paths
    all_paths = []
    
    # Helper function to recursively find all paths
    def find_all_paths(current_path: List[str]) -> None:
        current_node = current_path[-1]
        if current_node == goal:
            all_paths.append(current_path.copy())
            return
        for neighbor in graph.get(current_node, []):
            if neighbor not in current_path:
                current_path.append(neighbor)
                find_all_paths(current_path)
                current_path.pop()
                
    # Start the search
    find_all_paths([start])
    
    # Find the shortest path
    if all_paths:
        shortest_path = min(all_paths, key=len)
        return shortest_path
    else:
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
result = british_museum_search(example_graph, start_node, goal_node)
print(f"The shortest path from {start_node} to {goal_node} is: {result}")

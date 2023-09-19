from heapq import heappush, heappop
import preq
from typing import List, Tuple, Dict

def beam_search(graph: Dict[str, List[str]], start: str, goal: str, heuristic: Dict[str, int], w: int = 2) -> List[str]:
    """
    Beam Search Algorithm to find a path from start to goal in a graph based on heuristic values.
    
    Parameters:
    - graph: A dictionary representing the graph. Keys are node names, and values are lists of neighbors.
    - start: The start node.
    - goal: The goal node.
    - heuristic: A dictionary representing the heuristic values for each node.
    - w: The beam width.
    
    Returns:
    - A list representing a path from start to goal based on the heuristic values.
    """
    
    # Initialize the current nodes and paths
    current_nodes = [(heuristic[start], [start])]
    
    while current_nodes:
        new_nodes = []
        
        # Expand w most promising nodes
        for h, current_path in sorted(current_nodes)[:w]:
            current_node = current_path[-1]
            
            # If the goal is reached, return the path
            if current_node == goal:
                return current_path
            
            # Add the neighbors for exploration
            for neighbor in graph.get(current_node, []):
                if neighbor not in current_path:
                    new_path = current_path + [neighbor]
                    new_h = sum(heuristic.get(node, 0) for node in new_path)
                    heappush(new_nodes, (new_h, new_path))
                    
        current_nodes = new_nodes
    
    return None

# Example usage
result = beam_search(preq.example_graph, preq.start_node, preq.goal_node, preq.example_heuristic, w=2)
print(result)


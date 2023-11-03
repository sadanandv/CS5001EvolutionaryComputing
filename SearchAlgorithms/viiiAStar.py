import preq
from typing import List, Tuple, Dict
from heapq import heappush, heappop

def a_star_search(graph: Dict[str, List[str]], start: str, goal: str, heuristic: Dict[str, int]) -> List[str]:
    # Initialize a priority queue with the start node and its cost
    pq = [(heuristic[start], 0, [start])]
    
    while pq:
        # Get the path with the lowest total cost (heuristic + path length)
        f_cost, g_cost, current_path = heappop(pq)
        current_node = current_path[-1]
        
        # Check if the goal is reached
        if current_node == goal:
            return current_path
        
        # Add neighbors to the priority queue
        for neighbor in graph.get(current_node, []):
            if neighbor not in current_path:
                new_path = current_path + [neighbor]
                new_g_cost = g_cost + 1  # Assuming each step has a cost of 1
                new_f_cost = new_g_cost + heuristic.get(neighbor, 0)
                heappush(pq, (new_f_cost, new_g_cost, new_path))
                
    return None



# Example usage
print("A* Search:", a_star_search(preq.example_graph, preq.start_node, preq.goal_node, preq.example_heuristic))



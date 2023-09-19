import preq
from typing import List, Tuple, Dict
from heapq import heappush, heappop

def best_first_search(graph: Dict[str, List[str]], start: str, goal: str, heuristic: Dict[str, int]) -> List[str]:
    # Initialize a priority queue with the start node and its heuristic cost
    pq = [(heuristic[start], [start])]
    
    while pq:
        # Get the path with the lowest heuristic cost
        cost, current_path = heappop(pq)
        current_node = current_path[-1]
        
        # Check if the goal is reached
        if current_node == goal:
            return current_path
        
        # Add neighbors to the priority queue
        for neighbor in graph.get(current_node, []):
            if neighbor not in current_path:
                new_path = current_path + [neighbor]
                new_cost = heuristic.get(neighbor, 0)
                heappush(pq, (new_cost, new_path))
                
    return None

#Example Usage

print("Best First Search:", best_first_search(preq.example_graph, preq.start_node, preq.goal_node, preq.example_heuristic))
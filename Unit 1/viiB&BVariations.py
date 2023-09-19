import preq
from typing import List, Tuple, Dict
from heapq import heappush, heappop


def branch_and_bound(graph: Dict[str, List[str]], start: str, goal: str) -> List[str]:
    # Initialize a priority queue with the start node and its cost
    pq = [(0, [start])]
    
    while pq:
        # Get the path with the lowest cost
        cost, current_path = heappop(pq)
        current_node = current_path[-1]
        
        # Check if the goal is reached
        if current_node == goal:
            return current_path
        
        # Add neighbors to the priority queue
        for neighbor in graph.get(current_node, []):
            if neighbor not in current_path:
                new_path = current_path + [neighbor]
                new_cost = cost + 1  # Assuming each step has a cost of 1
                heappush(pq, (new_cost, new_path))
                
    return None


def branch_and_bound_with_extended_list(graph: Dict[str, List[str]], start: str, goal: str) -> List[str]:
    pq = [(0, [start])]
    visited = set()
    
    while pq:
        cost, current_path = heappop(pq)
        current_node = current_path[-1]
        
        if current_node == goal:
            return current_path
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor in graph.get(current_node, []):
            new_path = current_path + [neighbor]
            new_cost = cost + 1
            heappush(pq, (new_cost, new_path))
            
    return None


def branch_and_bound_heuristic(graph: Dict[str, List[str]], start: str, goal: str, heuristic: Dict[str, int]) -> List[str]:
    pq = [(heuristic[start], [start])]
    
    while pq:
        cost, current_path = heappop(pq)
        current_node = current_path[-1]
        
        if current_node == goal:
            return current_path
        
        for neighbor in graph.get(current_node, []):
            if neighbor not in current_path:
                new_path = current_path + [neighbor]
                new_cost = cost + heuristic.get(neighbor, 0)
                heappush(pq, (new_cost, new_path))
                
    return None


def branch_and_bound_with_extended_list_heuristic(graph: Dict[str, List[str]], start: str, goal: str, heuristic: Dict[str, int]) -> List[str]:
    pq = [(heuristic[start], [start])]
    visited = set()
    
    while pq:
        cost, current_path = heappop(pq)
        current_node = current_path[-1]
        
        if current_node == goal:
            return current_path
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor in graph.get(current_node, []):
            new_path = current_path + [neighbor]
            new_cost = cost + heuristic.get(neighbor, 0)
            heappush(pq, (new_cost, new_path))
            
    return None

# Example usage

example_graph, start_node, goal_node, example_heuristic = preq.example_graph, preq.start_node, preq.goal_node, preq.example_heuristic



print("Branch and Bound:", branch_and_bound(example_graph, start_node, goal_node))
print("Branch and Bound with Extended List:", branch_and_bound_with_extended_list(example_graph, start_node, goal_node))
print("Branch and Bound (Heuristic):", branch_and_bound_heuristic(example_graph, start_node, goal_node, example_heuristic))
print("Branch and Bound with Extended List (Heuristic):", branch_and_bound_with_extended_list_heuristic(example_graph, start_node, goal_node, example_heuristic))

from typing import List, Tuple, Dict
from collections import deque
from heapq import heappush, heappop

example_graph = {
    'A': ['B', 'C', 'E'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F', 'G'],
    'D': ['B'],
    'E': ['A', 'B', 'D'],
    'F': ['C'],
    'G': ['C']
}

example_heuristic = {
    'A': 3,
    'B': 2,
    'C': 1,
    'D': 4,
    'E': 3,
    'F': 2,
    'G': 0
}

start_node = 'A'
goal_node = 'G'

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


def oracle_algorithm(graph: Dict[str, List[str]], start: str, goal: str) -> List[str]:
    """
    Oracle Algorithm to find the shortest path from start to goal in a graph.
    
    Parameters:
    - graph: A dictionary representing the graph. Keys are node names, and values are lists of neighbors.
    - start: The start node.
    - goal: The goal node.
    
    Returns:
    - A list representing the shortest path from start to goal.
    """
    
    # For demonstration purposes, let's assume the Oracle knows the shortest path for our example graph
    known_shortest_paths = {
        ('A', 'G'): ['A', 'C', 'G']
    }
    
    return known_shortest_paths.get((start, goal), None)


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
4


def menu_driven_pipeline():
    # Define the algorithms in a dictionary for easy access
    algorithms = {
        '1': ('British Museum Search', british_museum_search),
        '2': ('Depth First Search (DFS)', depth_first_search),
        '3': ('Breadth First Search (BFS)', breadth_first_search),
        '4': ('Hill Climbing', hill_climbing),
        '5': ('Oracle', oracle_algorithm),
        '6': ('Beam Search', beam_search),
        '7': ('Branch and Bound', branch_and_bound),
        '8': ('Branch and Bound with Extended List', branch_and_bound_with_extended_list),
        '9': ('Branch and Bound (Heuristic)', branch_and_bound_heuristic),
        '10': ('Branch and Bound with Extended List (Heuristic)', branch_and_bound_with_extended_list_heuristic),
        '11': ('A* Search', a_star_search),
        '12': ('Best First Search', best_first_search)
    }
    
    while True:
        # Display menu
        print("\nPath Finding Algorithms:")
        for key, (name, _) in algorithms.items():
            print(f"{key}. {name}")
        print("0. Exit")
        
        # Get user choice
        choice = input("\nEnter the number of the algorithm you want to use: ").strip()
        
        # Exit condition
        if choice == '0':
            print("Exiting. Goodbye!")
            break
        
        # Validate choice and get algorithm function
        if choice not in algorithms:
            print("Invalid choice. Please try again.")
            continue
        
        algorithm_name, algorithm_func = algorithms[choice]
        
        # Get graph and nodes input from user
        # For simplicity, we are asking for a dictionary and lists as input (this can be enhanced for better user experience)
        try:
            graph = eval(input("\nEnter the graph as a dictionary (e.g., {'A': ['B', 'C'], 'B': ['A', 'D']}): "))
            start_node = input("Enter the start node: ").strip()
            goal_node = input("Enter the goal node: ").strip()
            
            # For heuristic-based algorithms, get heuristic values
            if "Heuristic" in algorithm_name or algorithm_name in ["Hill Climbing", "Beam Search", "A* Search", "Best First Search"]:
                heuristic = eval(input("\nEnter the heuristic values as a dictionary (e.g., {'A': 3, 'B': 2}): "))
                path = algorithm_func(graph, start_node, goal_node, heuristic)
            else:
                path = algorithm_func(graph, start_node, goal_node)
            
            # Display the result
            if path:
                print(f"\nPath found by {algorithm_name}: {path}")
            else:
                print(f"\nNo path found by {algorithm_name}.")
                
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")
            
        # Ask user if they want to try another algorithm
        another = input("\nWould you like to try another algorithm? (yes/no): ").strip().lower()
        if another != 'yes'or 'y' or 'Y' or 'YES' or 'Yes' or '1':
            print("Exiting. Goodbye!")
            break

# Run the menu-driven pipeline
menu_driven_pipeline()

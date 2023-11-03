from typing import List, Tuple, Dict
from collections import deque

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
from collections import deque, defaultdict
import heapq

# Romania map data (city connections and distances)
romania_map = {
    'Arad': [('Sibiu', 140), ('Timisoara', 118), ('Zerind', 75)],
    'Sibiu': [('Arad', 140), ('Fagaras', 99), ('Oradea', 151), ('Rimnicu Vilcea', 80)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Zerind': [('Arad', 75), ('Oradea', 71)],
    'Oradea': [('Sibiu', 151), ('Zerind', 71)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Pitesti', 97), ('Craiova', 146)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Craiova': [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85)],
    'Giurgiu': [('Bucharest', 90)],
    'Urziceni': [('Bucharest', 85), ('Hirsova', 98), ('Vaslui', 142)],
    'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
    'Eforie': [('Hirsova', 86)],
    'Vaslui': [('Urziceni', 142), ('Iasi', 92)],
    'Iasi': [('Vaslui', 92), ('Neamt', 87)],
    'Neamt': [('Iasi', 87)]
}

# Straight-line distances to Bucharest (heuristic)
heuristic = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
    'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151,
    'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
    'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193, 'Sibiu': 253,
    'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

def breadth_first_search(start, goal):
    """Breadth-First Search implementation."""
    print(f"\n=== Breadth-First Search: {start} to {goal} ===\n")
    
    frontier = deque([(start, [start], 0)])  # (node, path, cost)
    explored = set()
    step = 1
    
    while frontier:
        current, path, cost = frontier.popleft()
        
        print(f"Step {step}:")
        print(f"  Current: {current}")
        print(f"  Path: {' → '.join(path)}")
        print(f"  Cost: {cost}")
        print(f"  Frontier: {[node for node, _, _ in frontier]}")
        print(f"  Explored: {list(explored)}")
        print()
        
        if current == goal:
            print(f"GOAL FOUND! Path: {' → '.join(path)}")
            print(f"Total Distance: {cost} km")
            return path, cost
        
        explored.add(current)
        
        for neighbor, distance in romania_map[current]:
            if neighbor not in explored and neighbor not in [node for node, _, _ in frontier]:
                new_path = path + [neighbor]
                new_cost = cost + distance
                frontier.append((neighbor, new_path, new_cost))
        
        step += 1
    
    return None, None

def a_star_search(start, goal):
    """A* Search implementation."""
    print(f"\n=== A* Search: {start} to {goal} ===\n")
    
    # Priority queue: (f_score, g_score, current, path)
    frontier = [(heuristic[start], 0, start, [start])]
    explored = set()
    step = 1
    
    while frontier:
        f_score, g_score, current, path = heapq.heappop(frontier)
        
        print(f"Step {step}:")
        print(f"  Current: {current}")
        print(f"  Path: {' → '.join(path)}")
        print(f"  g({current}) = {g_score}, h({current}) = {heuristic[current]}, f({current}) = {f_score}")
        print(f"  Frontier: {[(f, node) for f, _, node, _ in frontier]}")
        print(f"  Explored: {list(explored)}")
        print()
        
        if current == goal:
            print(f"GOAL FOUND! Path: {' → '.join(path)}")
            print(f"Total Distance: {g_score} km")
            return path, g_score
        
        explored.add(current)
        
        for neighbor, distance in romania_map[current]:
            if neighbor not in explored:
                new_g_score = g_score + distance
                new_f_score = new_g_score + heuristic[neighbor]
                new_path = path + [neighbor]
                
                # Check if this path is better than any existing path to neighbor
                better_path = True
                for i, (f, g, node, p) in enumerate(frontier):
                    if node == neighbor and g <= new_g_score:
                        better_path = False
                        break
                
                if better_path:
                    heapq.heappush(frontier, (new_f_score, new_g_score, neighbor, new_path))
        
        step += 1
    
    return None, None

def print_romania_map():
    """Print the Romania map structure."""
    print("=== Romania Map Structure ===")
    for city, connections in romania_map.items():
        print(f"{city}: {', '.join([f'{neighbor}({dist}km)' for neighbor, dist in connections])}")
    print()

def print_heuristics():
    """Print heuristic values."""
    print("=== Straight-line Distances to Bucharest ===")
    for city, distance in sorted(heuristic.items()):
        print(f"{city}: {distance} km")
    print()

if __name__ == "__main__":
    print_romania_map()
    print_heuristics()
    
    # Run both search algorithms
    start_city = 'Arad'
    goal_city = 'Bucharest'
    
    # BFS
    bfs_path, bfs_cost = breadth_first_search(start_city, goal_city)
    
    # A*
    astar_path, astar_cost = a_star_search(start_city, goal_city)
    
    # Comparison
    print("\n=== Algorithm Comparison ===")
    print(f"BFS Path: {' → '.join(bfs_path)}")
    print(f"BFS Cost: {bfs_cost} km")
    print(f"A* Path: {' → '.join(astar_path)}")
    print(f"A* Cost: {astar_cost} km")
    
    if bfs_path == astar_path:
        print("Both algorithms found the same optimal path!")
    else:
        print("Algorithms found different paths.") 
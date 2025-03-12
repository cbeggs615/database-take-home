#!/usr/bin/env python3
import json
import os
import sys
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

# Add project root to path to import scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Import constants
from scripts.constants import (
    NUM_NODES,
    MAX_EDGES_PER_NODE,
    MAX_TOTAL_EDGES,
)


def load_graph(graph_file):
    """Load graph from a JSON file."""
    with open(graph_file, "r") as f:
        return json.load(f)


def load_results(results_file):
    """Load query results from a JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def save_graph(graph, output_file):
    """Save graph to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=2)


def verify_constraints(graph, max_edges_per_node, max_total_edges):
    """Verify that the graph meets all constraints."""
    # Check total edges
    total_edges = sum(len(edges) for edges in graph.values())
    if total_edges > max_total_edges:
        print(
            f"WARNING: Graph has {total_edges} edges, exceeding limit of {max_total_edges}"
        )
        return False

    # Check max edges per node
    max_node_edges = max(len(edges) for edges in graph.values())
    if max_node_edges > max_edges_per_node:
        print(
            f"WARNING: A node has {max_node_edges} edges, exceeding limit of {max_edges_per_node}"
        )
        return False

    # Check all nodes are present
    if len(graph) != NUM_NODES:
        print(f"WARNING: Graph has {len(graph)} nodes, should have {NUM_NODES}")
        return False

    # Check edge weights are valid (between 0 and 10)
    for node, edges in graph.items():
        for target, weight in edges.items():
            if weight <= 0 or weight > 10:
                print(f"WARNING: Edge {node} -> {target} has invalid weight {weight}")
                return False

    return True

def analyze_target_nodes(results, num_nodes=75):
    """
    Analyze and visualize node targeting from query results.

    Args:
        results: The query results data
        num_nodes: Total number of nodes in the graph

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    # Count how often each node is targeted
    target_freq = Counter()

    query_results = results.get('detailed_results', [])

    for query in query_results:
        target = query['target']
        target_freq[target] += 1

    # Convert to sorted list for easier visualization
    sorted_target_freq = sorted(target_freq.items(), key=lambda x: x[0])

    nodes = [item[0] for item in sorted_target_freq]
    frequencies = [item[1] for item in sorted_target_freq]

    # Visualize the target frequency
    plt.figure(figsize=(12, 6))
    plt.bar(nodes, frequencies, color='skyblue')
    plt.xlabel('Node ID')
    plt.ylabel('Target Frequency')
    plt.title('Node Targeting Frequency from Query Results')
    plt.xticks(range(0, num_nodes, max(1, num_nodes // 20)))  # Reduce ticks for large graphs
    plt.show()


def optimize_graph(
    initial_graph,
    results,
    num_nodes=NUM_NODES,
    max_total_edges=int(MAX_TOTAL_EDGES),
    max_edges_per_node=MAX_EDGES_PER_NODE,
):
    """
    Optimize the graph to improve random walk query performance.

    Args:
        initial_graph: Initial graph adjacency list
        results: Results from queries on the initial graph
        num_nodes: Number of nodes in the graph
        max_total_edges: Maximum total edges allowed
        max_edges_per_node: Maximum edges per node

    Returns:
        Optimized graph
    """
    print("Starting graph optimization...")

    from collections import Counter, defaultdict
    import heapq

    # Create a copy of the initial graph to modify

    optimized_graph = {node: {} for node in range(num_nodes)}
    # for node, edges in initial_graph.items():
    #     optimized_graph[node] = dict(edges)

    # =============================================================
    # TODO: Implement your optimization strategy here
    # =============================================================
    #
    # Your goal is to optimize the graph structure to:
    # 1. Increase the success rate of queries
    # 2. Minimize the path length for successful queries
    #
    # You have access to:
    # - initial_graph: The current graph structure
    # - results: The results of running queries on the initial graph
    #
    # Query results contain:
    # - Each query's target node
    # - Whether the query was successful
    # - The path taken during the random walk
    #
    # Remember the constraints:
    # - max_total_edges: Maximum number of edges in the graph
    # - max_edges_per_node: Maximum edges per node
    # - All nodes must remain in the graph
    # - Edge weights must be positive and ≤ 10

    # ---------------------------------------------------------------
    # EXAMPLE: Simple strategy to meet edge count constraint
    # This is just a basic example - you should implement a more
    # sophisticated strategy based on query analysis!
    # ---------------------------------------------------------------

    # Count total edges in the initial graph
    total_edges = sum(len(edges) for edges in optimized_graph.values())

    # analyze_target_nodes(results)

    target_freq = Counter()
    transitions_count = defaultdict(Counter)
    node_importance = defaultdict(float)

    query_results = results.get('detailed_results', [])


    for query in query_results:
        target = query['target']
        target_freq[target] +=1

        if query['is_success']:
            paths = query.get('paths', [])
            for path_info in paths:
                if path_info[0]: #successful
                    path = path_info[1]
                    for i in range(len(path)-1):
                        source, dest = path[i], path[i+1]
                        transitions_count[source][dest]+=1

                    # for last transition, give extra weight (directly connected to target)
                    if len(path)>=2 and path[-1]==target:
                        last_source = path[-2]
                        transitions_count[last_source][target]+=5



    # find node importance by seeing how often is was targeted
    total_queries = sum(target_freq.values())
    node_importance = {node: count/total_queries * 100 for node,count in target_freq.items()}

    valid_nodes = set(range(45))  # Nodes to prioritize

    edge_queue = []
    edge_count = 0
    out_edges = {node: 0 for node in range(num_nodes)}

    for source in range(num_nodes):
        for target in range(num_nodes):
            if source in valid_nodes and target in valid_nodes and source != target:
                utility = 0
                utility += transitions_count[source][target] * 2
                utility += node_importance.get(target, 0)
                utility += 0.01  # Baseline utility
                heapq.heappush(edge_queue, (-utility, source, target))

    # Build graph primarily for valid nodes
    while edge_queue and edge_count < max_total_edges:
        utility, source, target = heapq.heappop(edge_queue)
        if out_edges[source] >= max_edges_per_node:
            continue
        weight = min(10.0, 1.0 + 9.0 * min(1.0, (-utility) / 50))  # Weight based on utility
        optimized_graph[source][target] = weight
        out_edges[source] += 1
        edge_count += 1

    # Ensure nodes >= 45 have minimal connectivity if required
    for node in range(45, num_nodes):
        if out_edges[node] < max_edges_per_node and edge_count < max_total_edges:
            # Connect to valid nodes with high importance
            best_target = None
            best_score = -1

            for valid_node in valid_nodes:
                if valid_node not in optimized_graph[node]:
                    score = node_importance.get(valid_node, 0)
                    if score > best_score:
                        best_score = score
                        best_target = valid_node

            if best_target is not None:
                weight = 1.0 + 9.0 * (best_score / 100)  # Scale weight based on importance
                optimized_graph[node][best_target] = min(10.0, weight)  # Add edge
                out_edges[node] += 1
                edge_count += 1

    # if edge_count < max_total_edges:
    #     for source in valid_nodes:
    #         for target in valid_nodes:
    #             if source != target and target not in optimized_graph[source]:
    #                 utility = transitions_count[source][target] * 2
    #                 utility += node_importance.get(target, 0)
    #                 if utility > 0:  # Only add useful edges
    #                     weight = min(10.0, 1.0 + 9.0 * (utility / 50))
    #                     optimized_graph[source][target] = weight
    #                     out_edges[source] += 1
    #                     edge_count += 1
    #                     if edge_count >= max_total_edges:
    #                         break

    # # find node importance by seeing how often is was targeted
    # total_queries = sum(target_freq.values())
    # node_importance = {node: count/total_queries * 100 for node,count in target_freq.items()}


    # # priority queue of potential edges based on usefulness

    # edge_queue = []

    # for source in range(num_nodes):
    #     for target in range (num_nodes):
    #         if source == target :
    #             continue # no self loops

    #         utility = 0
    #         # utility if from how many times that edge was taken in successful paths
    #         utility += transitions_count[source][target] * 2

    #         # if target is important (more common), edge is more useful
    #         utility += node_importance.get(target, 0)

    #         # all edges have some utility
    #         utility += 0.01

    #         # min-heap of negative utility (to get highest utility first)
    #         heapq.heappush(edge_queue, (-utility, source, target))

    # # now build graph by adding most useful edges first
    # edge_count = 0
    # out_edges = {node: 0 for node in range(num_nodes)}

    # while edge_queue and edge_count < max_total_edges:
    #     utility, source, target = heapq.heappop(edge_queue)
    #     if out_edges[source] >= max_edges_per_node:
    #         continue

    #     raw_utility = -utility if utility < 0 else 0 #since it was entered as - in queue

    #     # find edge weight
    #     # higher edge weight means more likely to be used, so for most useful edges
    #     weight = min(10.0, 1.0 + 9.0 * min(1.0, raw_utility / 50))

    #     # Add edge to graph
    #     optimized_graph[source][target] = weight
    #     out_edges[source] += 1
    #     edge_count += 1

    # # all nodes have at least one outgoing edge for connectivity
    # for node in range(num_nodes):
    #     if out_edges[node] == 0 and edge_count < max_total_edges:
    #         # find best target based on importance
    #         best_target = None
    #         best_score = -1

    #         for potential_target in range(num_nodes):
    #             if potential_target != node:
    #                 score = node_importance.get(potential_target, 0)
    #                 if score > best_score:
    #                     best_score = score
    #                     best_target = potential_target

    #         # If no important targets found, connect to a random node
    #         if best_target is None:
    #             best_target = (node + 1) % num_nodes  # Avoid self-loops

    #         # Add edge with maximum weight for better connectivity
    #         optimized_graph[node][best_target] = 10.0
    #         out_edges[node] += 1
    #         edge_count += 1

    # # add connections to important nodes

    # if edge_count < max_total_edges:
    #     important_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
    #     # top 20 nodes
    #     top_nodes = [node for node, _ in important_nodes[:20]]



    #     # make heap for important connections
    #     addtl_edges = []
    #     for source in range (num_nodes):
    #         if out_edges[source] < max_edges_per_node:
    #             for target in top_nodes:
    #                 if source != target and target not in optimized_graph[source]:
    #                     # priority based on target importance
    #                     priority = node_importance.get(target, 0)
    #                     heapq.heappush(addtl_edges, (-priority, source, target))

    #     while addtl_edges and edge_count<max_total_edges:
    #         _, source, target = heapq.heappop(addtl_edges)
    #         if out_edges[source] < max_edges_per_node and target not in optimized_graph[source]:
    #             optimized_graph[source][target] = 10.0  # max weight for important connections
    #             out_edges[source] += 1
    #             edge_count += 1

    # =============================================================
    # End of your implementation
    # =============================================================

    # Verify constraints
    if not verify_constraints(optimized_graph, max_edges_per_node, max_total_edges):
        print("WARNING: Your optimized graph does not meet the constraints!")
        print("The evaluation script will reject it. Please fix the issues.")

    return optimized_graph


if __name__ == "__main__":
    # Get file paths
    initial_graph_file = os.path.join(project_dir, "data", "initial_graph.json")
    results_file = os.path.join(project_dir, "data", "initial_results.json")
    output_file = os.path.join(
        project_dir, "candidate_submission", "optimized_graph.json"
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading initial graph from {initial_graph_file}")
    initial_graph = load_graph(initial_graph_file)

    print(f"Loading query results from {results_file}")
    results = load_results(results_file)

    print("Optimizing graph...")
    optimized_graph = optimize_graph(initial_graph, results)

    print(f"Saving optimized graph to {output_file}")
    save_graph(optimized_graph, output_file)

    print("Done! Optimized graph has been saved.")

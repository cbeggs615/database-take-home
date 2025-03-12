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

    query_results = results.get('detailed_results', [])

    for query in query_results:
        target = query['target']
        target_freq[target] +=1

    important_nodes = sorted(target_freq.keys(), key=lambda node: target_freq[node], reverse=True)
    targets = 45
    actually_targeted = [node for node in important_nodes if node < targets]

    optimized_graph = {node: {} for node in range(num_nodes)}

    # Create first cycle of connections
    for i in range(len(actually_targeted)):
        current = actually_targeted[i]
        next_node = actually_targeted[(i + 1) % len(actually_targeted)] #wraps
        optimized_graph[current][next_node] = 1

    # Minimal connectivity for other nodes
    for node in range(num_nodes):
        if node >= targets:
            optimized_graph[node][actually_targeted[0]] = 1

    optimized_graph[actually_targeted[-1]][actually_targeted[0]] = 4


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

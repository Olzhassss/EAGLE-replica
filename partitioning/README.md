# Partitioning Module

This module provides functionality for partitioning graphs into subgraphs using the Balanced Louvain algorithm. It is designed to work with graph data and includes visualization capabilities for the resulting partitions.

## Requirements

- Python 3.x
- Required Python packages (install via `pip`):
  - `networkx`
  - `matplotlib`

### Dependencies

- **`benchmark_consts`**: Provides the `get_args` function to parse arguments.
- **`problem`**: Contains the `Problem` class for handling graph data.
- **`balanced_louvain`**: Implements the Balanced Louvain algorithm for graph partitioning.
- **`vis`**: Provides visualization utilities for graph partitions.

## How to Use

1. **Topology File**:
   - Add support for your topology in `benchmark_consts` and add the file in the graphml file in the `topology-zoo` subdirectory.

2. **Run the Script**:
   - Execute the script from the command line:
     ```bash
     python partitioning.py --topo example_topology.json
     ```
   - Provide the necessary arguments as required by the `get_args` function.

3. **View Results**:
   - The script will partition the graph and visualize the results using the `vis_partitions` function.

## Notes

- The Balanced Louvain algorithm parameters (e.g., `resolution`, `gamma`, `epsilon`, `rho`, `seed`) are hardcoded in the `benchmark` function. You can modify these values to experiment with different partitioning results.
- Ensure all dependencies are installed and accessible in the Python environment.


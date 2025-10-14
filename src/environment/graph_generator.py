"""
Graph generator for creating product flow environments.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import networkx as nx


@dataclass
class GraphConfig:
    """Configuration for graph generation."""

    num_nodes: int = 10
    branching_factor: float = 2.0  # Average number of branches per node
    max_depth: int = 5  # Maximum depth of the graph
    popup_probability: float = 0.1
    dead_end_probability: float = 0.1
    dependency_probability: float = 0.1
    num_dependencies: int = 2
    random_seed: int = 42


@dataclass
class DistributionShiftConfig:
    """Configuration for creating distribution shift variants."""

    # Depth shift: increase graph depth
    depth_multiplier: float = 1.0  # Multiply depth by this factor

    # Complexity shift: increase branching
    branching_multiplier: float = 1.0  # Multiply branching factor by this

    # Size shift: increase number of nodes
    size_multiplier: float = 1.0  # Multiply num_nodes by this

    # Topology shift: change graph structure
    topology_type: Optional[str] = None  # 'linear', 'wide', 'deep', None for default


class GraphGenerator:
    """
    Generates directed graphs representing product flows.

    Graphs have a start node, goal node, and intermediate nodes with various
    action types on edges. Ensures at least one valid path from start to goal.

    Supports train/test splits and distribution shift variants for
    generalization evaluation.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the graph generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

    def generate(self, config: GraphConfig) -> nx.DiGraph:
        """
        Generate a directed graph based on configuration.

        Args:
            config: Graph configuration specifying structure parameters

        Returns:
            Directed graph with annotated nodes and edges
        """
        if config.random_seed is not None:
            random.seed(config.random_seed)

        graph = nx.DiGraph()

        # Step 1: Create backbone path from start to goal
        backbone_length = min(config.max_depth, config.num_nodes)
        backbone_nodes = list(range(backbone_length))

        # Add backbone nodes with labels
        for i, node_id in enumerate(backbone_nodes):
            if i == 0:
                node_type = "start"
            elif i == backbone_length - 1:
                node_type = "goal"
            else:
                node_type = "intermediate"

            graph.add_node(node_id, node_type=node_type)

        # Add backbone edges with action labels
        action_types = ["click", "type", "navigate"]
        for i in range(len(backbone_nodes) - 1):
            action = random.choice(action_types)
            graph.add_edge(backbone_nodes[i], backbone_nodes[i + 1], action_type=action)

        # Step 2: Add additional nodes and branches
        next_node_id = backbone_length
        remaining_nodes = config.num_nodes - backbone_length

        # Distribute remaining nodes across backbone nodes for branching
        if remaining_nodes > 0 and backbone_length > 1:
            # Exclude start and goal from branching sources
            branch_sources = backbone_nodes[:-1]  # All except goal

            for _ in range(remaining_nodes):
                if not branch_sources:
                    break

                # Select a source node for branching based on branching factor
                # Higher branching factor = more likely to branch
                if random.random() < config.branching_factor:
                    source_node = random.choice(branch_sources)

                    # Create new branch node
                    new_node_id = next_node_id
                    next_node_id += 1

                    # Determine if this is a dead-end or connects back
                    is_dead_end = random.random() < 0.3  # 30% chance of dead-end

                    if is_dead_end:
                        node_type = "dead-end"
                        graph.add_node(new_node_id, node_type=node_type)
                        action = random.choice(action_types)
                        graph.add_edge(source_node, new_node_id, action_type=action)
                    else:
                        node_type = "intermediate"
                        graph.add_node(new_node_id, node_type=node_type)
                        action = random.choice(action_types)
                        graph.add_edge(source_node, new_node_id, action_type=action)

                        # Connect back to backbone or another node
                        # Find valid target nodes (nodes after source in backbone)
                        source_idx = (
                            backbone_nodes.index(source_node)
                            if source_node in backbone_nodes
                            else 0
                        )
                        valid_targets = [
                            n
                            for n in backbone_nodes
                            if backbone_nodes.index(n) > source_idx
                        ]

                        if valid_targets:
                            target_node = random.choice(valid_targets)
                            action = random.choice(action_types)
                            graph.add_edge(new_node_id, target_node, action_type=action)

        # Step 3: Add some back edges for complexity (optional loops)
        if config.branching_factor > 0.5 and len(graph.nodes) > 3:
            num_back_edges = int(config.branching_factor * len(graph.nodes) * 0.1)
            for _ in range(num_back_edges):
                nodes_list = list(graph.nodes)
                source = random.choice(nodes_list)
                target = random.choice(nodes_list)

                # Avoid self-loops and edges to start node
                if source != target and graph.nodes[target].get("node_type") != "start":
                    if not graph.has_edge(source, target):
                        action = random.choice(action_types)
                        graph.add_edge(source, target, action_type=action)

        # Verify at least one path exists from start to goal
        start_node = backbone_nodes[0]
        goal_node = backbone_nodes[-1]

        if not nx.has_path(graph, start_node, goal_node):
            # This shouldn't happen with backbone, but ensure it
            # by adding direct edge if needed
            action = random.choice(action_types)
            graph.add_edge(start_node, goal_node, action_type=action)

        return graph

    def generate_train_test_split(
        self,
        config: GraphConfig,
        num_train: int,
        num_test: int,
        train_seed_offset: int = 0,
        test_seed_offset: int = 10000,
    ) -> Tuple[List[nx.DiGraph], List[nx.DiGraph]]:
        """
        Generate train and test graph sets with distinct random seeds.

        This ensures test graphs are different from training graphs while
        maintaining similar structural properties.

        Args:
            config: Base graph configuration
            num_train: Number of training graphs to generate
            num_test: Number of test graphs to generate
            train_seed_offset: Starting seed for training graphs
            test_seed_offset: Starting seed for test graphs (should be >> train_seed_offset)

        Returns:
            Tuple of (train_graphs, test_graphs)
        """
        train_graphs = []
        test_graphs = []

        # Generate training graphs
        for i in range(num_train):
            train_config = GraphConfig(
                num_nodes=config.num_nodes,
                branching_factor=config.branching_factor,
                max_depth=config.max_depth,
                random_seed=train_seed_offset + i,
            )
            train_graph = self.generate(train_config)
            train_graphs.append(train_graph)

        # Generate test graphs with different seeds
        for i in range(num_test):
            test_config = GraphConfig(
                num_nodes=config.num_nodes,
                branching_factor=config.branching_factor,
                max_depth=config.max_depth,
                random_seed=test_seed_offset + i,
            )
            test_graph = self.generate(test_config)
            test_graphs.append(test_graph)

        return train_graphs, test_graphs

    def generate_distribution_shift(
        self,
        base_config: GraphConfig,
        shift_config: DistributionShiftConfig,
        num_graphs: int,
        seed_offset: int = 20000,
    ) -> List[nx.DiGraph]:
        """
        Generate graphs with distribution shift for out-of-distribution evaluation.

        Creates variants with different properties (deeper, wider, different topology)
        to test generalization capabilities.

        Args:
            base_config: Base graph configuration
            shift_config: Configuration specifying the type and magnitude of shift
            num_graphs: Number of shifted graphs to generate
            seed_offset: Starting seed for shifted graphs

        Returns:
            List of graphs with distribution shift applied
        """
        shifted_graphs = []

        for i in range(num_graphs):
            # Apply distribution shift to config
            shifted_config = self._apply_distribution_shift(
                base_config, shift_config, seed_offset + i
            )

            # Generate graph with shifted configuration
            graph = self.generate(shifted_config)
            shifted_graphs.append(graph)

        return shifted_graphs

    def _apply_distribution_shift(
        self,
        base_config: GraphConfig,
        shift_config: DistributionShiftConfig,
        seed: int,
    ) -> GraphConfig:
        """
        Apply distribution shift to a base configuration.

        Args:
            base_config: Base graph configuration
            shift_config: Shift parameters
            seed: Random seed for this graph

        Returns:
            Modified GraphConfig with shift applied
        """
        # Apply multipliers
        new_depth = int(base_config.max_depth * shift_config.depth_multiplier)
        new_branching = base_config.branching_factor * shift_config.branching_multiplier
        new_num_nodes = int(base_config.num_nodes * shift_config.size_multiplier)

        # Ensure minimum values
        new_depth = max(new_depth, 2)
        new_branching = max(new_branching, 0.1)
        new_num_nodes = max(new_num_nodes, new_depth)

        # Apply topology shift if specified
        if shift_config.topology_type == "linear":
            # Linear topology: minimal branching, maximum depth
            new_branching = 0.1
            new_depth = new_num_nodes
        elif shift_config.topology_type == "wide":
            # Wide topology: high branching, shallow depth
            new_branching = min(new_branching * 2.0, 2.0)
            new_depth = max(new_depth // 2, 2)
        elif shift_config.topology_type == "deep":
            # Deep topology: low branching, maximum depth
            new_branching = max(new_branching * 0.5, 0.1)
            new_depth = int(new_depth * 1.5)

        return GraphConfig(
            num_nodes=new_num_nodes,
            branching_factor=new_branching,
            max_depth=new_depth,
            random_seed=seed,
        )

    def create_standard_splits(
        self,
        base_config: GraphConfig,
        num_train: int = 50,
        num_test_in_dist: int = 20,
        num_test_per_shift: int = 10,
    ) -> Dict[str, List[nx.DiGraph]]:
        """
        Create standard train/test splits with multiple distribution shifts.

        This is a convenience method that creates a complete evaluation setup:
        - Training set
        - In-distribution test set
        - Multiple out-of-distribution test sets with different shifts

        Args:
            base_config: Base graph configuration
            num_train: Number of training graphs
            num_test_in_dist: Number of in-distribution test graphs
            num_test_per_shift: Number of graphs per distribution shift variant

        Returns:
            Dictionary mapping split names to lists of graphs:
            - 'train': Training graphs
            - 'test_in_dist': In-distribution test graphs
            - 'test_deeper': Deeper graphs (depth shift)
            - 'test_wider': Wider graphs (branching shift)
            - 'test_larger': Larger graphs (size shift)
            - 'test_linear': Linear topology
            - 'test_wide_topology': Wide topology
        """
        # Generate train and in-distribution test sets
        train_graphs, test_in_dist_graphs = self.generate_train_test_split(
            config=base_config,
            num_train=num_train,
            num_test=num_test_in_dist,
            train_seed_offset=0,
            test_seed_offset=10000,
        )

        # Generate distribution shift variants
        # 1. Deeper graphs (1.5x depth)
        deeper_shift = DistributionShiftConfig(depth_multiplier=1.5)
        test_deeper = self.generate_distribution_shift(
            base_config, deeper_shift, num_test_per_shift, seed_offset=20000
        )

        # 2. Wider graphs (1.5x branching)
        wider_shift = DistributionShiftConfig(branching_multiplier=1.5)
        test_wider = self.generate_distribution_shift(
            base_config, wider_shift, num_test_per_shift, seed_offset=21000
        )

        # 3. Larger graphs (1.5x size)
        larger_shift = DistributionShiftConfig(size_multiplier=1.5)
        test_larger = self.generate_distribution_shift(
            base_config, larger_shift, num_test_per_shift, seed_offset=22000
        )

        # 4. Linear topology
        linear_shift = DistributionShiftConfig(topology_type="linear")
        test_linear = self.generate_distribution_shift(
            base_config, linear_shift, num_test_per_shift, seed_offset=23000
        )

        # 5. Wide topology
        wide_topology_shift = DistributionShiftConfig(topology_type="wide")
        test_wide_topology = self.generate_distribution_shift(
            base_config, wide_topology_shift, num_test_per_shift, seed_offset=24000
        )

        return {
            "train": train_graphs,
            "test_in_dist": test_in_dist_graphs,
            "test_deeper": test_deeper,
            "test_wider": test_wider,
            "test_larger": test_larger,
            "test_linear": test_linear,
            "test_wide_topology": test_wide_topology,
        }

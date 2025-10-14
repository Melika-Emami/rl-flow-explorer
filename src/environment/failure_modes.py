"""
Failure mode classes for product flow environments.

This module defines different types of failures that can occur in product flows:
- Dead-ends: States with no valid exit
- Stochastic pop-ups: Random pop-ups that mask actions
- Hidden dependencies: Earlier actions that modify later transitions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import random
import networkx as nx


@dataclass
class FailureModeConfig:
    """Base configuration for failure modes."""

    enabled: bool = True
    random_seed: Optional[int] = None


@dataclass
class DeadEndConfig(FailureModeConfig):
    """Configuration for dead-end failure mode."""

    num_dead_ends: int = 0  # Number of dead-end nodes to create
    dead_end_probability: float = (
        0.0  # Probability of marking existing nodes as dead-ends
    )


@dataclass
class PopupConfig(FailureModeConfig):
    """Configuration for stochastic pop-up failure mode."""

    popup_probability: float = 0.1  # Probability of pop-up appearing
    affected_nodes: Optional[List[int]] = None  # Specific nodes affected, None = all


@dataclass
class DependencyConfig(FailureModeConfig):
    """Configuration for hidden dependency failure mode."""

    num_dependencies: int = 1  # Number of dependency chains to create
    dependency_length: int = 2  # Length of each dependency chain


class FailureMode(ABC):
    """Abstract base class for failure modes."""

    def __init__(self, config: FailureModeConfig):
        """
        Initialize failure mode.

        Args:
            config: Configuration for this failure mode
        """
        self.config = config
        if config.random_seed is not None:
            random.seed(config.random_seed)

    @abstractmethod
    def inject(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Inject this failure mode into the graph.

        Args:
            graph: The graph to modify

        Returns:
            Modified graph with failure mode injected
        """
        pass

    @abstractmethod
    def check_failure(
        self, current_node: int, action: int, next_node: int, state: Dict
    ) -> Tuple[bool, str]:
        """
        Check if this failure mode is triggered.

        Args:
            current_node: Current node ID
            action: Action taken
            next_node: Next node ID
            state: Current environment state

        Returns:
            Tuple of (is_failure, failure_message)
        """
        pass


class DeadEndFailure(FailureMode):
    """
    Dead-end failure mode.

    Marks nodes with no outgoing edges (except possibly back navigation).
    When an agent reaches a dead-end, it cannot progress to the goal.
    """

    def __init__(self, config: DeadEndConfig):
        """
        Initialize dead-end failure mode.

        Args:
            config: Configuration for dead-end failures
        """
        super().__init__(config)
        self.config: DeadEndConfig = config
        self.dead_end_nodes: Set[int] = set()

    def inject(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Inject dead-end nodes into the graph.

        This marks existing nodes as dead-ends by removing their outgoing edges
        (except back edges) or identifies nodes that are already dead-ends.

        Args:
            graph: The graph to modify

        Returns:
            Modified graph with dead-end nodes marked
        """
        if not self.config.enabled:
            return graph

        # Find nodes that are already dead-ends (no outgoing edges)
        existing_dead_ends = [
            node
            for node in graph.nodes()
            if graph.out_degree(node) == 0
            and graph.nodes[node].get("node_type") != "goal"
        ]

        # Mark existing dead-ends
        for node in existing_dead_ends:
            graph.nodes[node]["node_type"] = "dead-end"
            graph.nodes[node]["failure_mode"] = "dead_end"
            self.dead_end_nodes.add(node)

        # Create additional dead-ends if specified
        if self.config.num_dead_ends > 0:
            # Get intermediate nodes that aren't already dead-ends
            candidates = [
                node
                for node in graph.nodes()
                if graph.nodes[node].get("node_type") == "intermediate"
                and node not in self.dead_end_nodes
            ]

            # Select nodes to convert to dead-ends
            num_to_create = min(self.config.num_dead_ends, len(candidates))
            if num_to_create > 0:
                selected = random.sample(candidates, num_to_create)

                for node in selected:
                    # Remove all outgoing edges
                    outgoing_edges = list(graph.out_edges(node))
                    for edge in outgoing_edges:
                        graph.remove_edge(*edge)

                    # Mark as dead-end
                    graph.nodes[node]["node_type"] = "dead-end"
                    graph.nodes[node]["failure_mode"] = "dead_end"
                    self.dead_end_nodes.add(node)

        # Apply probability-based dead-end marking
        if self.config.dead_end_probability > 0:
            candidates = [
                node
                for node in graph.nodes()
                if graph.nodes[node].get("node_type") == "intermediate"
                and node not in self.dead_end_nodes
            ]

            for node in candidates:
                if random.random() < self.config.dead_end_probability:
                    # Remove all outgoing edges
                    outgoing_edges = list(graph.out_edges(node))
                    for edge in outgoing_edges:
                        graph.remove_edge(*edge)

                    # Mark as dead-end
                    graph.nodes[node]["node_type"] = "dead-end"
                    graph.nodes[node]["failure_mode"] = "dead_end"
                    self.dead_end_nodes.add(node)

        return graph

    def check_failure(
        self, current_node: int, action: int, next_node: int, state: Dict
    ) -> Tuple[bool, str]:
        """
        Check if agent has reached a dead-end.

        Args:
            current_node: Current node ID
            action: Action taken (unused)
            next_node: Next node ID (unused)
            state: Current environment state (unused)

        Returns:
            Tuple of (is_failure, failure_message)
        """
        if current_node in self.dead_end_nodes:
            return True, f"Reached dead-end node {current_node}"
        return False, ""


class StochasticPopupFailure(FailureMode):
    """
    Stochastic pop-up failure mode.

    Random pop-ups appear with configurable probability, masking available actions
    and potentially causing the agent to fail if it cannot handle them.
    """

    def __init__(self, config: PopupConfig):
        """
        Initialize stochastic pop-up failure mode.

        Args:
            config: Configuration for pop-up failures
        """
        super().__init__(config)
        self.config: PopupConfig = config
        self.affected_nodes: Set[int] = set()

    def inject(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Inject pop-up failure mode into the graph.

        Marks nodes where pop-ups can appear. The actual pop-up occurrence
        is stochastic and checked during environment execution.

        Args:
            graph: The graph to modify

        Returns:
            Modified graph with pop-up annotations
        """
        if not self.config.enabled:
            return graph

        # Determine which nodes can have pop-ups
        if self.config.affected_nodes is not None:
            # Use specified nodes
            self.affected_nodes = set(self.config.affected_nodes)
        else:
            # All intermediate nodes can have pop-ups
            self.affected_nodes = {
                node
                for node in graph.nodes()
                if graph.nodes[node].get("node_type") in ["intermediate", "start"]
            }

        # Annotate nodes with pop-up capability
        for node in self.affected_nodes:
            graph.nodes[node]["popup_enabled"] = True
            graph.nodes[node]["popup_probability"] = self.config.popup_probability

        return graph

    def check_failure(
        self, current_node: int, action: int, next_node: int, state: Dict
    ) -> Tuple[bool, str]:
        """
        Check if a pop-up appears and causes failure.

        Pop-ups appear stochastically. When they appear, they mask actions
        and the agent must handle them correctly (implementation detail
        left to environment).

        Args:
            current_node: Current node ID
            action: Action taken
            next_node: Next node ID
            state: Current environment state (should contain 'popup_active' flag)

        Returns:
            Tuple of (is_failure, failure_message)
        """
        # Check if pop-up is currently active
        if state.get("popup_active", False):
            # Pop-up is active - this could be a failure depending on action
            # For now, we consider it a failure if the agent tries to take
            # a normal action while pop-up is active
            return True, f"Pop-up active at node {current_node}, action blocked"

        return False, ""

    def should_trigger_popup(self, node: int) -> bool:
        """
        Determine if a pop-up should appear at this node.

        Args:
            node: Node ID to check

        Returns:
            True if pop-up should appear
        """
        if node in self.affected_nodes:
            return random.random() < self.config.popup_probability
        return False


class HiddenDependencyFailure(FailureMode):
    """
    Hidden dependency failure mode.

    Earlier actions in the flow silently modify later state transitions.
    For example, skipping a required form field early in the flow causes
    a failure later when trying to submit.
    """

    def __init__(self, config: DependencyConfig):
        """
        Initialize hidden dependency failure mode.

        Args:
            config: Configuration for dependency failures
        """
        super().__init__(config)
        self.config: DependencyConfig = config
        # Maps (trigger_node, action) -> (check_node, required_state)
        self.dependencies: Dict[Tuple[int, int], Tuple[int, str]] = {}
        # Maps dependency_id -> (trigger_node, action, check_node)
        self.dependency_chains: List[Tuple[int, int, int]] = []

    def inject(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Inject hidden dependencies into the graph.

        Creates dependency chains where an action at one node affects
        the outcome at a later node.

        Args:
            graph: The graph to modify

        Returns:
            Modified graph with dependency annotations
        """
        if not self.config.enabled:
            return graph

        # Get all nodes in topological order (if possible)
        try:
            node_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Graph has cycles, use arbitrary order
            node_order = list(graph.nodes())

        # Create dependency chains
        for dep_id in range(self.config.num_dependencies):
            # Select trigger node (early in flow)
            # and check node (later in flow)
            if len(node_order) < 2:
                continue

            # Trigger should be in first half, check in second half
            mid_point = len(node_order) // 2
            trigger_candidates = (
                node_order[:mid_point] if mid_point > 0 else node_order[:1]
            )
            check_candidates = (
                node_order[mid_point:]
                if mid_point < len(node_order)
                else node_order[-1:]
            )

            if not trigger_candidates or not check_candidates:
                continue

            trigger_node = random.choice(trigger_candidates)
            check_node = random.choice(check_candidates)

            # Ensure there's a path from trigger to check
            if not nx.has_path(graph, trigger_node, check_node):
                continue

            # Get an action from trigger node
            outgoing_edges = list(graph.out_edges(trigger_node, data=True))
            if not outgoing_edges:
                continue

            # Select an edge/action
            edge_idx = random.randint(0, len(outgoing_edges) - 1)
            _, _, edge_data = outgoing_edges[edge_idx]

            # Create dependency
            dependency_key = f"dep_{dep_id}"
            self.dependencies[(trigger_node, edge_idx)] = (check_node, dependency_key)
            self.dependency_chains.append((trigger_node, edge_idx, check_node))

            # Annotate graph
            if "dependencies" not in graph.nodes[trigger_node]:
                graph.nodes[trigger_node]["dependencies"] = []
            graph.nodes[trigger_node]["dependencies"].append(
                {
                    "action": edge_idx,
                    "check_node": check_node,
                    "dependency_id": dependency_key,
                }
            )

            if "dependency_checks" not in graph.nodes[check_node]:
                graph.nodes[check_node]["dependency_checks"] = []
            graph.nodes[check_node]["dependency_checks"].append(
                {
                    "trigger_node": trigger_node,
                    "dependency_id": dependency_key,
                    "required": True,
                }
            )

        return graph

    def check_failure(
        self, current_node: int, action: int, next_node: int, state: Dict
    ) -> Tuple[bool, str]:
        """
        Check if a hidden dependency causes failure.

        Checks if the agent is at a check node and whether required
        dependencies were satisfied earlier in the episode.

        Args:
            current_node: Current node ID
            action: Action taken
            next_node: Next node ID
            state: Current environment state (should contain 'satisfied_dependencies')

        Returns:
            Tuple of (is_failure, failure_message)
        """
        # Check if current node has dependency checks
        satisfied_deps = state.get("satisfied_dependencies", set())

        # Look for dependencies that should be checked at this node
        for (trigger_node, trigger_action), (
            check_node,
            dep_id,
        ) in self.dependencies.items():
            if check_node == current_node:
                # This node requires a dependency
                if dep_id not in satisfied_deps:
                    return (
                        True,
                        f"Hidden dependency not satisfied: {dep_id} (required action at node {trigger_node})",
                    )

        return False, ""

    def record_action(self, node: int, action: int, state: Dict) -> Dict:
        """
        Record an action and update satisfied dependencies.

        Args:
            node: Current node ID
            action: Action taken
            state: Current environment state

        Returns:
            Updated state with satisfied dependencies
        """
        if "satisfied_dependencies" not in state:
            state["satisfied_dependencies"] = set()

        # Check if this action satisfies a dependency
        if (node, action) in self.dependencies:
            _, dep_id = self.dependencies[(node, action)]
            state["satisfied_dependencies"].add(dep_id)

        return state


class FailureModeInjector:
    """
    Utility class to inject multiple failure modes into a graph.
    """

    def __init__(self, failure_modes: List[FailureMode]):
        """
        Initialize the injector with a list of failure modes.

        Args:
            failure_modes: List of failure mode instances to inject
        """
        self.failure_modes = failure_modes

    def inject_all(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Inject all failure modes into the graph.

        Args:
            graph: The graph to modify

        Returns:
            Modified graph with all failure modes injected
        """
        for failure_mode in self.failure_modes:
            graph = failure_mode.inject(graph)

        return graph

    def check_failures(
        self, current_node: int, action: int, next_node: int, state: Dict
    ) -> List[Tuple[bool, str]]:
        """
        Check all failure modes for failures.

        Args:
            current_node: Current node ID
            action: Action taken
            next_node: Next node ID
            state: Current environment state

        Returns:
            List of (is_failure, failure_message) tuples for each failure mode
        """
        results = []
        for failure_mode in self.failure_modes:
            result = failure_mode.check_failure(current_node, action, next_node, state)
            results.append(result)

        return results

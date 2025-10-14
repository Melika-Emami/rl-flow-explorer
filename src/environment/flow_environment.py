"""
FlowEnvironment: Gym-compatible environment for product flow exploration.

This module implements a reinforcement learning environment where agents
navigate through product flows represented as directed graphs with various
failure modes and partial observability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

from .failure_modes import FailureMode, FailureModeInjector


@dataclass
class EnvConfig:
    """Configuration for FlowEnvironment."""

    step_penalty: float = -0.01  # Penalty for each step taken
    success_reward: float = 1.0  # Reward for reaching goal
    failure_penalty: float = -1.0  # Penalty for triggering failure
    max_steps: int = 100  # Maximum steps per episode
    partial_observability: bool = True  # Limit observation to local info
    random_seed: Optional[int] = None
    fixed_max_actions: Optional[int] = (
        None  # Fixed max actions for consistent obs space
    )


class FlowEnvironment(gym.Env):
    """
    Gym-compatible environment for product flow exploration.

    The environment represents a product flow as a directed graph where:
    - Nodes represent screens/states
    - Edges represent actions (click, type, navigate)
    - Various failure modes can be injected
    - Observations provide partial information about current state
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        graph: nx.DiGraph,
        config: EnvConfig,
        failure_modes: Optional[List[FailureMode]] = None,
    ):
        """
        Initialize the FlowEnvironment.

        Args:
            graph: Directed graph representing the product flow
            config: Environment configuration
            failure_modes: List of failure modes to inject (optional)
        """
        super().__init__()

        self.graph = graph
        self.config = config
        self.failure_modes = failure_modes or []
        self.failure_injector = (
            FailureModeInjector(self.failure_modes) if self.failure_modes else None
        )

        # Set random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Identify special nodes
        self.start_node = self._find_start_node()
        self.goal_node = self._find_goal_node()

        # Build action mapping: maps action index to (source_node, target_node, edge_data)
        self._build_action_space()

        # Build observation space
        self._build_observation_space()

        # Episode state
        self.current_node: Optional[int] = None
        self.step_count: int = 0
        self.episode_done: bool = False
        self.visited_nodes: set = set()
        self.visit_counts: Dict[int, int] = {}  # Track visit count per node

        # Hidden state for failure modes
        self.hidden_state: Dict[str, Any] = {}

    def _find_start_node(self) -> int:
        """Find the start node in the graph."""
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "start":
                return node
        # If no start node labeled, use node 0
        return 0

    def _find_goal_node(self) -> int:
        """Find the goal node in the graph."""
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "goal":
                return node
        # If no goal node labeled, use last node
        return max(self.graph.nodes())

    def _build_action_space(self):
        """Build the action space and action mapping."""
        # Use fixed max_actions if provided, otherwise compute from graph
        if self.config.fixed_max_actions is not None:
            max_actions = self.config.fixed_max_actions
        else:
            # Count maximum number of outgoing edges from any node
            max_actions = max(
                (self.graph.out_degree(node) for node in self.graph.nodes()), default=1
            )

        # Ensure at least 1 action
        max_actions = max(max_actions, 1)

        # Action space is discrete
        self.action_space = spaces.Discrete(max_actions)
        self.max_actions = max_actions

        # Build mapping from (node, action_idx) to target node
        self.action_map: Dict[Tuple[int, int], int] = {}

        for node in self.graph.nodes():
            outgoing_edges = list(self.graph.out_edges(node, data=True))
            for action_idx, (source, target, edge_data) in enumerate(outgoing_edges):
                self.action_map[(node, action_idx)] = target

    def _build_observation_space(self):
        """Build the observation space."""
        num_nodes = len(self.graph.nodes())

        # Observation components:
        # 1. One-hot encoding of current node
        # 2. Available actions mask
        # 3. Node features (node type, visit count)

        obs_size = (
            num_nodes  # One-hot current node
            + self.max_actions  # Available actions mask
            + 4  # Node features: is_start, is_goal, is_dead_end, visit_count_normalized
        )

        # Observation space is a box with values in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.num_nodes = num_nodes

    def _get_available_actions(self, node: int) -> np.ndarray:
        """
        Get mask of available actions from current node.

        Args:
            node: Current node ID

        Returns:
            Binary mask of available actions
        """
        mask = np.zeros(self.max_actions, dtype=np.float32)

        outgoing_edges = list(self.graph.out_edges(node))
        for action_idx in range(min(len(outgoing_edges), self.max_actions)):
            mask[action_idx] = 1.0

        return mask

    def _encode_observation(self, node: int) -> np.ndarray:
        """
        Encode the current state as an observation.

        Args:
            node: Current node ID

        Returns:
            Observation vector
        """
        # One-hot encoding of current node
        node_one_hot = np.zeros(self.num_nodes, dtype=np.float32)
        node_one_hot[node] = 1.0

        # Available actions mask
        actions_mask = self._get_available_actions(node)

        # Node features
        node_data = self.graph.nodes[node]
        node_type = node_data.get("node_type", "intermediate")

        is_start = 1.0 if node_type == "start" else 0.0
        is_goal = 1.0 if node_type == "goal" else 0.0
        is_dead_end = 1.0 if node_type == "dead-end" else 0.0

        # Normalized visit count (cap at 10 for normalization)
        visit_count = self.visit_counts.get(node, 0)
        visit_count_normalized = min(visit_count / 10.0, 1.0)

        node_features = np.array(
            [is_start, is_goal, is_dead_end, visit_count_normalized], dtype=np.float32
        )

        # Concatenate all components
        observation = np.concatenate([node_one_hot, actions_mask, node_features])

        return observation

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset episode state
        self.current_node = self.start_node
        self.step_count = 0
        self.episode_done = False
        self.visited_nodes = {self.start_node}
        self.visit_counts = {self.start_node: 1}

        # Reset hidden state for failure modes
        self.hidden_state = {
            "satisfied_dependencies": set(),
            "popup_active": False,
        }

        # Get initial observation
        observation = self._encode_observation(self.current_node)

        info = {
            "current_node": self.current_node,
            "step_count": self.step_count,
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute an action and return the result.

        Args:
            action: Action index to execute

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.step_count += 1
        reward = self.config.step_penalty  # Default step penalty
        terminated = False
        truncated = False
        info = {}

        # Check if action is valid
        if (self.current_node, action) not in self.action_map:
            # Invalid action - stay in same state, apply penalty
            reward = self.config.failure_penalty
            info["failure_type"] = "invalid_action"
            info["message"] = f"Invalid action {action} at node {self.current_node}"

            observation = self._encode_observation(self.current_node)
            return observation, reward, terminated, truncated, info

        # Get next node
        next_node = self.action_map[(self.current_node, action)]

        # Update hidden state for dependencies
        if self.failure_injector:
            for failure_mode in self.failure_modes:
                if hasattr(failure_mode, "record_action"):
                    self.hidden_state = failure_mode.record_action(
                        self.current_node, action, self.hidden_state
                    )

        # Check for failures before transition
        failure_occurred = False
        if self.failure_injector:
            failure_results = self.failure_injector.check_failures(
                self.current_node, action, next_node, self.hidden_state
            )

            for is_failure, failure_message in failure_results:
                if is_failure:
                    failure_occurred = True
                    reward = self.config.failure_penalty
                    terminated = True
                    info["failure_type"] = "failure_mode"
                    info["message"] = failure_message
                    break

        # Check for stochastic pop-ups
        if not failure_occurred and self.failure_injector:
            for failure_mode in self.failure_modes:
                if hasattr(failure_mode, "should_trigger_popup"):
                    if failure_mode.should_trigger_popup(next_node):
                        self.hidden_state["popup_active"] = True
                        # Pop-up doesn't immediately fail, but affects next step
                        info["popup_triggered"] = True
                    else:
                        self.hidden_state["popup_active"] = False

        # Transition to next node if no failure
        if not failure_occurred:
            self.current_node = next_node
            self.visited_nodes.add(next_node)
            self.visit_counts[next_node] = self.visit_counts.get(next_node, 0) + 1

            # Check if reached goal
            if self.current_node == self.goal_node:
                reward = self.config.success_reward
                terminated = True
                info["success"] = True
                info["message"] = "Reached goal node"

            # Check if reached dead-end
            elif self.graph.nodes[self.current_node].get("node_type") == "dead-end":
                # Dead-end check happens on arrival
                if self.graph.out_degree(self.current_node) == 0:
                    reward = self.config.failure_penalty
                    terminated = True
                    info["failure_type"] = "dead_end"
                    info["message"] = f"Reached dead-end node {self.current_node}"

        # Check for timeout
        if self.step_count >= self.config.max_steps:
            truncated = True
            info["timeout"] = True
            info["message"] = "Episode timeout"

        # Mark episode as done
        if terminated or truncated:
            self.episode_done = True

        # Get observation
        observation = self._encode_observation(self.current_node)

        # Add additional info
        info["current_node"] = self.current_node
        info["step_count"] = self.step_count
        info["visited_nodes"] = len(self.visited_nodes)

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current state of the environment.

        Args:
            mode: Rendering mode ('human' or 'ansi')

        Returns:
            String representation if mode='ansi', None otherwise
        """
        output = []
        output.append(f"Step: {self.step_count}/{self.config.max_steps}")
        output.append(f"Current Node: {self.current_node}")
        output.append(
            f"Node Type: {self.graph.nodes[self.current_node].get('node_type', 'unknown')}"
        )
        output.append(f"Visited Nodes: {len(self.visited_nodes)}/{self.num_nodes}")

        # Show available actions
        available_actions = []
        for action_idx in range(self.max_actions):
            if (self.current_node, action_idx) in self.action_map:
                target = self.action_map[(self.current_node, action_idx)]
                available_actions.append(f"  {action_idx} -> Node {target}")

        if available_actions:
            output.append("Available Actions:")
            output.extend(available_actions)
        else:
            output.append("No available actions (dead-end)")

        output_str = "\n".join(output)

        if mode == "human":
            print(output_str)
            return None
        elif mode == "ansi":
            return output_str
        else:
            return None

    def close(self):
        """Clean up resources."""
        pass

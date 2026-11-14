"""Swarm: peer-to-peer federated learning with no central server."""
from .model import MLP
from .node import Node
from .data import make_dataset, split_non_iid

__all__ = ["MLP", "Node", "make_dataset", "split_non_iid"]

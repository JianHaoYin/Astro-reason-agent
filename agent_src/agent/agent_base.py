from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from message_manger import MessageManager
from logger import Logger
from model import BaseModel


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        *args,
        model: BaseModel,
        message_manager: MessageManager,
        logger: Logger,
        max_turns: int,
        **kwargs,
    ) -> None:
        """
        Initialize the agent.

        Args:
            model: The underlying model used by the agent.
            message_manager: Component responsible for managing messages.
            logger: Logger instance for recording runtime information.
            max_turns: Maximum number of interaction turns.
        """
        self.model: BaseModel = model
        self.message_manager: MessageManager = message_manager
        self.logger: Logger = logger
        self.max_turns: int = max_turns

    @abstractmethod
    def run(self, sandbox_dir: Path, output_dir: Path) -> dict[str, Any]:
        """
        Run the agent.

        Args:
            sandbox_dir: Directory for intermediate or sandbox files.
            output_dir: Directory for final outputs.

        Returns:
            A dictionary containing execution results.
        """
        raise NotImplementedError
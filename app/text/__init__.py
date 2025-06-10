"""Text splitter interfaces."""

from abc import ABC, abstractmethod
from typing import List

class TextSplitter(ABC):
    """Interface for text splitters."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""

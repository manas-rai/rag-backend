from abc import ABC, abstractmethod
from typing import List, Dict, Any

class TextSplitter(ABC):
    """Interface for text splitting strategies."""
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the splitter."""
        pass

class TextProcessor(ABC):
    """Interface for text processing."""
    
    @abstractmethod
    def process_text(self, text: str) -> str:
        """Process text (e.g., cleaning, normalization)."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the processor."""
        pass 
"""Prompt manager for handling RAG system prompts."""

from pathlib import Path
from typing import Dict, Any
from app.utils.logger import setup_logger

logger = setup_logger('prompt_manager')

class PromptManager:
    """Manages loading and formatting of prompts for the RAG system."""

    def __init__(self):
        """Initialize the prompt manager."""
        self.base_dir = Path(__file__).parent
        self.prompts: Dict[str, Dict[str, str]] = {
            "system": {},
            "user": {}
        }
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompt templates from the templates directories."""
        try:
            # Load system prompts
            system_dir = self.base_dir / "system"
            for template_file in system_dir.glob("*.txt"):
                prompt_name = template_file.stem
                with open(template_file, "r") as f:
                    self.prompts["system"][prompt_name] = f.read()

            # Load user prompts
            user_dir = self.base_dir / "user"
            for template_file in user_dir.glob("*.txt"):
                prompt_name = template_file.stem
                with open(template_file, "r") as f:
                    self.prompts["user"][prompt_name] = f.read()

            logger.info(
                "Loaded prompts: %d system, %d user",
                len(self.prompts["system"]),
                len(self.prompts["user"])
            )
        except Exception as e:
            logger.error("Error loading prompt templates: %s", str(e))
            raise

    def format_prompt(
        self,
        prompt_type: str,
        prompt_name: str,
        context: str,
        query: str,
        **kwargs: Any
    ) -> str:
        """Format a prompt template with the given parameters.
        
        Args:
            prompt_type: Type of prompt ("system", "user", or "tenant")
            prompt_name: Name of the prompt template to use
            context: The retrieved context from the vector store
            query: The user's query
            **kwargs: Additional parameters to format the prompt
            
        Returns:
            Formatted prompt string
        """
        try:
            if prompt_type not in self.prompts:
                raise ValueError(f"Invalid prompt type: {prompt_type}")

            else:
                if prompt_name not in self.prompts[prompt_type]:
                    raise ValueError(f"Prompt template '{prompt_name}' not found in {prompt_type}")
                template = self.prompts[prompt_type][prompt_name]

            # Format the context into a readable string
            formatted_context = self._format_context(context)

            # Get the template and format it
            formatted_prompt = template.format(
                context=formatted_context,
                query=query,
                **kwargs
            )

            logger.debug(
                "Formatted %s prompt '%s'",
                prompt_type,
                prompt_name
            )
            return formatted_prompt
        except Exception as e:
            logger.error("Error formatting prompt: %s", str(e))
            raise

    def _format_context(self, context: str) -> str:
        """Format the context into a readable string.
        
        Args:
            context: The raw context from the vector store
            
        Returns:
            Formatted context string
        """
        # Split context into chunks and format each chunk
        chunks = context.split("\n\n")
        formatted_chunks = []

        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():
                formatted_chunks.append(f"Chunk {i}:\n{chunk.strip()}")

        return "\n\n".join(formatted_chunks)

    def get_available_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Get a dictionary of all available prompts.
        
        Returns:
            Dictionary containing all available prompts by type and name
        """
        return {
            "system": list(self.prompts["system"].keys()),
            "user": list(self.prompts["user"].keys())
        }

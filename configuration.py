import os
from pydantic import BaseModel, Field
from typing import Any, Optional, List
from langchain_core.runnables import RunnableConfig

class Configuration(BaseModel):
    # LLM Configuration
    query_generator_model: str = Field(
        default="microsoft/DialoGPT-medium",  # Default to a smaller model
        metadata={
            "description": "The Hugging Face model name for query generation."
        }
    )

    reflection_model: str = Field(
        default="microsoft/DialoGPT-medium",
        metadata={
            "description": "The Hugging Face model name for reflection."
        }
    )

    answer_model: str = Field(
        default="microsoft/DialoGPT-medium",
        metadata={
            "description": "The Hugging Face model name for answer generation."
        }
    )

    # Search Configuration
    search_engines: List[str] = Field(
        default=["bing", "google", "kaggle", "nih", "openneuro"],
        metadata={
            "description": "List of search engines to use."
        }
    )

    # API Keys
    bing_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Bing Search API key."
        }
    )

    kaggle_username: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Kaggle username for API access."
        }
    )

    kaggle_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Kaggle API key."
        }
    )

    # Hugging Face Configuration
    huggingface_token: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Hugging Face API token for accessing private/protected models."
        }
    )

    # Model Configuration
    max_tokens: int = Field(
        default=512,
        metadata={
            "description": "Maximum tokens for model generation."
        }
    )

    temperature: float = Field(
        default=0.7,
        metadata={
            "description": "Temperature for model generation."
        }
    )

    # Research Configuration
    number_of_initial_queries: int = Field(
        default=5,
        metadata={
            "description": "The number of initial search queries to generate.",
        }
    )

    max_research_loops: int = Field(
        default=2,
        metadata={
            "description": "The maximum number of research loops to perform."
        }
    )

    # Citation Configuration
    citation_prefix: str = Field(
        default="https://neurosearch.id/",
        metadata={
            "description": "URL prefix for citation links in generated text."
        }
    )

    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
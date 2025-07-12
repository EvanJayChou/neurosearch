from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why those queries are relevant to the research topic."
    )

class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )

class DatasetInfo(BaseModel):
    name: str = Field(description="Name of the dataset.")
    type: str = Field(description="Type of dataset, e.g., EEG, MEG, MRI, etc.")
    features: Dict[str, str] = Field(description="Key features of the dataset, e.g., number of subjects, sampling rate.")
    metadata: Dict[str, str] = Field(description="Metadata such as age range, gender, diagnosis, acquisition parameters.")
    access: Optional[str] = Field(default=None, description="Access requirements, e.g., open, restricted.")
    license: Optional[str] = Field(default=None, description="License information.")
    publication: Optional[str] = Field(default=None, description="Publication reference.")
    source: Optional[str] = Field(default=None, description="Source URL or citation.")
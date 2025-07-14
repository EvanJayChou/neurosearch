from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator

class DatasetInfo(TypedDict):
    name: str
    type: str  # EEG, MEG, MRI, etc.
    features: Dict[str, Any]  # e.g., {"subjects": 100, "sampling_rate": "500Hz"}
    metadata: Dict[str, Any]  # e.g., {"age_range": "18-65", "gender": "mixed"}
    access: str  # e.g., "open", "restricted"
    license: str
    publication: str
    source: str  # URL or citation
    download_url: Optional[str]
    file_size: Optional[str]
    format: Optional[str]  # e.g., "BIDS", "EEGLAB", "MNE"

class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str
    source: str  # "bing", "google", "kaggle", "nih", "openneuro"
    metadata: Dict[str, Any]

class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    search_results: Annotated[list, operator.add]  # New: structured search results
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    datasets_extracted: List[DatasetInfo]
    search_engines_used: List[str]

class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int

class Query(TypedDict):
    query: str
    rationale: str
    target_sources: List[str]  # Which search engines to use

class QueryGenerationState(TypedDict):
    search_query: List[Query]

class WebSearchState(TypedDict):
    search_query: str
    id: str
    target_sources: List[str]
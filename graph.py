import os
import json
from typing import List, Dict, Any, Optional

from tools import SearchQueryList, Reflection, DatasetInfo
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

# Hugging Face imports
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from configuration import Configuration
from prompts import (
    get_current_date,
    query_writer_instructions,
    dataset_searcher_instructions,  # updated name
    reflection_instructions,
    answer_instructions,
)
from utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from search_tools import MultiSearchEngine, SearchResult

load_dotenv()

def create_huggingface_llm(model_name: str, config: Configuration):
    """Create a Hugging Face LLM pipeline."""
    try:
        # Set up authentication if token is provided
        auth_kwargs = {}
        if config.huggingface_token:
            auth_kwargs["token"] = config.huggingface_token
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        
        # If it's an authentication error and we don't have a token, suggest getting one
        if "401" in str(e) or "authentication" in str(e).lower():
            print(f"Authentication required for {model_name}. Please:")
            print("1. Get a Hugging Face token from https://huggingface.co/settings/tokens")
            print("2. Add HUGGINGFACE_TOKEN=your_token to your .env file")
            print("3. Or use a public model like 'microsoft/DialoGPT-medium'")
        
        # Fallback to a smaller public model
        if model_name != "microsoft/DialoGPT-medium":
            print("Falling back to microsoft/DialoGPT-medium...")
            return create_huggingface_llm("microsoft/DialoGPT-medium", config)
        else:
            raise e

# Initialize search engine
search_engine = None

def initialize_search_engine(config: Configuration):
    """Initialize the multi-search engine."""
    global search_engine
    if search_engine is None:
        search_engine = MultiSearchEngine(config)
    return search_engine

# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries using Hugging Face LLM."""
    configurable = Configuration.from_runnable_config(config)
    
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Initialize Hugging Face LLM
    llm = create_huggingface_llm(configurable.query_generator_model, configurable)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
        search_engines=", ".join(configurable.search_engines)
    )
    
    # Generate queries (simplified for Hugging Face models)
    response = llm.invoke(formatted_prompt)
    
    # Parse response to extract queries (you might need to adjust this based on your model)
    queries = parse_queries_from_response(response, configurable.search_engines)
    
    return {"search_query": queries}

def parse_reflection_response(response: str) -> Dict[str, Any]:
    """Parse reflection response from Hugging Face model output."""
    try:
        # Simple parsing for Hugging Face models
        # Look for keywords in the response
        response_lower = response.lower()
        
        # Check if sufficient
        is_sufficient = any(keyword in response_lower for keyword in 
                          ['sufficient', 'complete', 'enough', 'adequate'])
        
        # Extract knowledge gap
        knowledge_gap = ""
        if 'gap' in response_lower or 'missing' in response_lower:
            # Try to extract the gap description
            lines = response.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['gap', 'missing', 'need']):
                    knowledge_gap = line.strip()
                    break
        
        # Extract follow-up queries
        follow_up_queries = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # Simple heuristic: lines that look like questions
                if line.endswith('?') or any(keyword in line.lower() for keyword in 
                                           ['what', 'how', 'why', 'when', 'where', 'which']):
                    follow_up_queries.append(line)
        
        return {
            "is_sufficient": is_sufficient,
            "knowledge_gap": knowledge_gap,
            "follow_up_queries": follow_up_queries[:3]  # Limit to 3 queries
        }
    except Exception as e:
        print(f"Error parsing reflection response: {e}")
        return {
            "is_sufficient": False,
            "knowledge_gap": "Unable to parse response",
            "follow_up_queries": []
        }

def parse_queries_from_response(response: str, available_engines: List[str]) -> List[Dict[str, Any]]:
    """Parse queries from LLM response and assign target sources."""
    # This is a simplified parser - you might want to use structured output or better parsing
    lines = response.strip().split('\n')
    queries = []
    
    for line in lines:
        if line.strip() and not line.startswith('#'):
            # Assign all queries to neuroscience dataset APIs
            target_sources = ['kaggle', 'nih', 'openneuro']
            queries.append({
                "query": line.strip(),
                "rationale": f"Generated query for {', '.join(target_sources)}",
                "target_sources": target_sources
            })
    
    return queries[:5]  # Limit to 5 queries

def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node."""
    return [
        Send("web_research", {
            "search_query": query["query"], 
            "id": int(idx),
            "target_sources": query["target_sources"]
        })
        for idx, query in enumerate(state["search_query"])
    ]

def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs multi-source web research."""
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize search engine
    search_engine = initialize_search_engine(configurable)
    
    # Perform search across multiple sources
    search_results = search_engine.search(
        state["search_query"], 
        state["target_sources"], 
        count=10
    )
    
    # Process results and extract datasets
    datasets_extracted = []
    processed_results = []
    
    for result in search_results:
        processed_results.append({
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "source": result.source,
            "metadata": result.metadata
        })
        
        # Extract dataset information if available
        if result.source in ['kaggle', 'nih', 'openneuro']:
            dataset_info = extract_dataset_info(result)
            if dataset_info:
                datasets_extracted.append(dataset_info)
    
    # Generate summary using Hugging Face LLM
    llm = create_huggingface_llm(configurable.query_generator_model, configurable)
    
    formatted_prompt = dataset_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
        search_results=json.dumps(processed_results, indent=2)
    )
    
    summary = llm.invoke(formatted_prompt)
    
    # Add citations to the summary
    resolved_urls_map = resolve_urls(search_results, state["id"], configurable.citation_prefix)
    citations = get_citations(search_results, resolved_urls_map)
    summary_with_citations = insert_citation_markers(summary, citations)
    
    return {
        "sources_gathered": processed_results,
        "search_query": [state["search_query"]],
        "web_research_result": [summary_with_citations],
        "datasets_extracted": datasets_extracted,
        "search_results": processed_results
    }

def extract_dataset_info(result: SearchResult) -> Optional[DatasetInfo]:
    """Extract structured dataset information from search results."""
    try:
        metadata = result.metadata
        
        # Extract common fields
        dataset_info = {
            "name": result.title,
            "type": infer_dataset_type(result.title, result.snippet),
            "features": extract_features(metadata),
            "metadata": metadata,
            "access": metadata.get("access", "unknown"),
            "license": metadata.get("license", "unknown"),
            "publication": metadata.get("publication", ""),
            "source": result.url,
            "download_url": metadata.get("download_url"),
            "file_size": metadata.get("size"),
            "format": infer_format(result.title, metadata)
        }
        
        return dataset_info
    except Exception as e:
        print(f"Error extracting dataset info: {e}")
        return None

def infer_dataset_type(title: str, snippet: str) -> str:
    """Infer dataset type from title and snippet."""
    text = (title + " " + snippet).lower()
    
    if any(keyword in text for keyword in ['eeg', 'electroencephalography']):
        return "EEG"
    elif any(keyword in text for keyword in ['meg', 'magnetoencephalography']):
        return "MEG"
    elif any(keyword in text for keyword in ['mri', 'fmri', 'functional']):
        return "MRI"
    elif any(keyword in text for keyword in ['pet', 'positron']):
        return "PET"
    else:
        return "Unknown"

def extract_features(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key features from metadata."""
    features = {}
    
    # Common feature keys
    feature_keys = ['subjects', 'participants', 'samples', 'size', 'files']
    
    for key in feature_keys:
        if key in metadata:
            features[key] = metadata[key]
    
    return features

def infer_format(title: str, metadata: Dict[str, Any]) -> str:
    """Infer data format from title and metadata."""
    text = title.lower()
    
    if 'bids' in text:
        return "BIDS"
    elif 'eeglab' in text or '.set' in text:
        return "EEGLAB"
    elif 'mne' in text or '.fif' in text:
        return "MNE"
    else:
        return "Unknown"


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = create_huggingface_llm(reasoning_model, configurable)
    result = llm.invoke(formatted_prompt)

    # Parse the result to extract structured information
    # For Hugging Face models, we need to parse the text response
    parsed_result = parse_reflection_response(result)
    
    return {
        "is_sufficient": parsed_result.get("is_sufficient", False),
        "knowledge_gap": parsed_result.get("knowledge_gap", ""),
        "follow_up_queries": parsed_result.get("follow_up_queries", []),
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                    "target_sources": ["kaggle", "nih", "openneuro"],
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model
    llm = create_huggingface_llm(reasoning_model, configurable)
    result = llm.invoke(formatted_prompt)

    # Add citations to the final answer
    all_search_results = []
    for search_result in state.get("search_results", []):
        all_search_results.extend(search_result)
    
    if all_search_results:
        resolved_urls_map = resolve_urls(all_search_results, 0, configurable.citation_prefix)
        citations = get_citations(all_search_results, resolved_urls_map)
        result_with_citations = insert_citation_markers(result, citations)
    else:
        result_with_citations = result

    return {
        "messages": [AIMessage(content=result_with_citations)],
        "sources_gathered": state.get("sources_gathered", []),
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set up the graph edges
builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="neurosearch-agent")
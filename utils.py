from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(search_results: List[Any], id: int, citation_prefix: str = "https://neurosearch.id/") -> Dict[str, str]:
    """
    Create a map of search result URLs to short URLs with a unique id for each url.
    Ensures each original URL gets a consistent shortened form while maintaining uniqueness.
    
    Args:
        search_results: List of search result objects
        id: Unique identifier for this search session
        citation_prefix: URL prefix for citation links (default: "https://neurosearch.id/")
    """
    prefix = citation_prefix
    
    # Extract URLs from search results
    urls = []
    for result in search_results:
        if hasattr(result, 'url'):
            urls.append(result.url)
        elif isinstance(result, dict) and 'url' in result:
            urls.append(result['url'])

    # Create a dictionary that maps each unique URL to its first occurrence index
    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text: str, citations_list: List[Dict[str, Any]]) -> str:
    """
    Inserts citation markers into a text string at the end of sentences or paragraphs.

    Args:
        text (str): The original text string.
        citations_list (list): A list of citation dictionaries with segments.

    Returns:
        str: The text with citation markers inserted.
    """
    if not citations_list:
        return text
    
    # Create a list of all citation markers
    all_markers = []
    for citation in citations_list:
        for segment in citation.get("segments", []):
            marker = f" [{segment['label']}]({segment['short_url']})"
            all_markers.append(marker)
    
    # Add citations at the end of the text
    if all_markers:
        citation_text = "\n\n**Sources:**" + "".join(all_markers)
        return text + citation_text
    
    return text


def get_citations(search_results: List[Any], resolved_urls_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Creates citation information from search results for use in text generation.

    This function processes search results to create citation objects that can be
    used to add references to generated text. Each citation includes information
    about the source and can be used to create markdown links.

    Args:
        search_results: List of search result objects from various search engines
        resolved_urls_map: Dictionary mapping original URLs to shortened URLs

    Returns:
        list: A list of dictionaries, where each dictionary represents a citation
              and has the following keys:
              - "segments" (list): List of citation segments with label, short_url, and value
    """
    citations = []
    
    for result in search_results:
        citation = {
            "segments": []
        }
        
        # Extract URL and title from search result
        url = None
        title = None
        
        if hasattr(result, 'url'):
            url = result.url
            title = getattr(result, 'title', 'Unknown Source')
        elif isinstance(result, dict):
            url = result.get('url')
            title = result.get('title', 'Unknown Source')
        
        if url and url in resolved_urls_map:
            short_url = resolved_urls_map[url]
            
            # Create a clean label from the title
            if title:
                # Remove common file extensions and clean up the title
                clean_title = title.split('.')[0]  # Remove file extension
                clean_title = clean_title.replace('_', ' ').replace('-', ' ')
                label = clean_title[:50]  # Limit length
            else:
                label = f"Source {len(citations) + 1}"
            
            citation["segments"].append({
                "label": label,
                "short_url": short_url,
                "value": url,
            })
            
            citations.append(citation)
    
    return citations
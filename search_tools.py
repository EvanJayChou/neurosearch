import requests
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import kaggle
from urllib.parse import urljoin, quote
import time

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    metadata: Dict[str, Any]

class BingSearchTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    def search(self, query: str, count: int = 10) -> List[SearchResult]:
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": count,
            "mkt": "en-US",
            "responseFilter": "Webpages"
        }
        
        try:
            response = requests.get(self.endpoint, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="bing",
                    metadata={"rank": item.get("position", 0)}
                ))
            return results
        except Exception as e:
            print(f"Bing search error: {e}")
            return []

class KaggleSearchTool:
    def __init__(self, username: str, key: str):
        self.username = username
        self.key = key
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
    
    def search(self, query: str, count: int = 10) -> List[SearchResult]:
        try:
            # Search for datasets
            datasets = kaggle.api.dataset_list(search=query, max_results=count)
            
            results = []
            for dataset in datasets:
                # Get detailed info
                dataset_info = kaggle.api.dataset_view(dataset.ref)
                
                results.append(SearchResult(
                    title=dataset.title,
                    url=f"https://www.kaggle.com/datasets/{dataset.ref}",
                    snippet=dataset.subtitle or "",
                    source="kaggle",
                    metadata={
                        "size": dataset.size,
                        "downloads": dataset.downloads,
                        "votes": dataset.votes,
                        "license": dataset.licenseName,
                        "tags": dataset.tags
                    }
                ))
            return results
        except Exception as e:
            print(f"Kaggle search error: {e}")
            return []

class NIHSearchTool:
    def __init__(self):
        self.base_url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
    
    def search(self, query: str, count: int = 10) -> List[SearchResult]:
        try:
            # Search for datasets in NIH repositories
            url = f"{self.base_url}/search"
            params = {
                "query": query,
                "limit": count,
                "include": "dataset"
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("datasets", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("accession", ""),
                    snippet=item.get("description", ""),
                    source="nih",
                    metadata={
                        "accession": item.get("accession"),
                        "organism": item.get("organism", {}).get("name"),
                        "file_count": item.get("file_count")
                    }
                ))
            return results
        except Exception as e:
            print(f"NIH search error: {e}")
            return []

class OpenNeuroSearchTool:
    def __init__(self):
        self.base_url = "https://openneuro.org/api"
    
    def search(self, query: str, count: int = 10) -> List[SearchResult]:
        try:
            url = f"{self.base_url}/datasets"
            params = {
                "search": query,
                "limit": count
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data:
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=f"https://openneuro.org/datasets/{item.get('id')}",
                    snippet=item.get("description", ""),
                    source="openneuro",
                    metadata={
                        "id": item.get("id"),
                        "subjects": item.get("subjects"),
                        "modalities": item.get("modalities"),
                        "license": item.get("license")
                    }
                ))
            return results
        except Exception as e:
            print(f"OpenNeuro search error: {e}")
            return []

class MultiSearchEngine:
    def __init__(self, config):
        self.tools = {}
        
        if "bing" in config.search_engines and config.bing_api_key:
            self.tools["bing"] = BingSearchTool(config.bing_api_key)
        
        if "kaggle" in config.search_engines and config.kaggle_username and config.kaggle_key:
            self.tools["kaggle"] = KaggleSearchTool(config.kaggle_username, config.kaggle_key)
        
        if "nih" in config.search_engines:
            self.tools["nih"] = NIHSearchTool()
        
        if "openneuro" in config.search_engines:
            self.tools["openneuro"] = OpenNeuroSearchTool()
    
    def search(self, query: str, sources: List[str], count: int = 10) -> List[SearchResult]:
        all_results = []
        
        for source in sources:
            if source in self.tools:
                try:
                    results = self.tools[source].search(query, count)
                    all_results.extend(results)
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"Error searching {source}: {e}")
        
        return all_results 
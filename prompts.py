import datetime as datetime

def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

# === QUERY WRITER INSTRUCTIONS ===
query_writer_instructions = """
Your goal is to generate sophisticated and diverse web search queries to identify neuroscience datasets (such as EEG, MEG, MRI, fMRI, PET, etc.), their features, metadata, and any other useful information about each dataset. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Focus on finding publicly available neuroscience datasets, including but not limited to EEG, MEG, MRI, fMRI, PET, and related modalities.
- For each dataset, aim to extract information about its features (e.g., number of subjects, data modalities, sampling rates, experimental paradigms, etc.), metadata (e.g., age, gender, diagnosis, acquisition parameters), and any other relevant details (e.g., licensing, access requirements, publication references).
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: Find open-access EEG and MRI datasets with detailed metadata for cognitive neuroscience research
```json
{{
    "rationale": "To identify suitable datasets, we need to search for open-access EEG and MRI repositories, focusing on those that provide comprehensive metadata and documentation. These queries target sources and summaries of available datasets, as well as their features and access conditions.",
    "query": [
        "open-access EEG datasets with metadata and experimental details",
        "public MRI datasets for cognitive neuroscience with subject demographics and acquisition parameters",
        "list of neuroscience datasets with features, modalities, and access requirements"
    ]
}}
```

Context: {research_topic}
"""

# === WEB SEARCHER INSTRUCTIONS ===
web_searcher_instructions = """
Conduct targeted Google Searches to gather the most recent, credible information on neuroscience datasets related to "{research_topic}". Extract details about each dataset, including its type (EEG, MEG, MRI, etc.), features, metadata, and any other useful information (such as licensing, access requirements, and publication references).

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information about available neuroscience datasets.
- For each dataset found, extract and summarize:
    - Dataset type (EEG, MEG, MRI, fMRI, PET, etc.)
    - Features (number of subjects, modalities, sampling rates, experimental paradigms, etc.)
    - Metadata (subject demographics, acquisition parameters, diagnosis, etc.)
    - Access requirements, licensing, and publication references
    - Any other relevant or useful information
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""

# === REFLECTION INSTRUCTIONS ===
reflection_instructions = """
You are an expert research assistant analyzing summaries about neuroscience datasets for "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration regarding available neuroscience datasets, their features, metadata, or access conditions, and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding of available datasets or their characteristics.
- Focus on technical details, dataset completeness, metadata richness, or emerging datasets that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": false,
    "knowledge_gap": "The summary lacks information about the sampling rates and experimental paradigms used in the listed EEG datasets",
    "follow_up_queries": ["What are the sampling rates and experimental paradigms for the major open-access EEG datasets?"]
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

# === ANSWER INSTRUCTIONS ===
answer_instructions = """
Generate a high-quality answer to the user's question about neuroscience datasets based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- For each dataset, clearly describe its type (EEG, MEG, MRI, etc.), features, metadata, and any other useful information (such as licensing, access requirements, and publication references).
- Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [apnews](https://vertexaisearch.cloud.google.com/id/1-0)). THIS IS A MUST.

User Context:
- {research_topic}

Summaries:
{summaries}
"""
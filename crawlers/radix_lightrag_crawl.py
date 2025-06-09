import asyncio
import os
import httpx
import requests
from xml.etree import ElementTree
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from dotenv import load_dotenv
# from agents.lightrag_react_agent import embedding_func, llm_model_func
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from lightrag.utils import EmbeddingFunc
from lightrag.llm.azure_openai import azure_openai_complete_if_cache, azure_openai_embed
import numpy as np

# Load environment variables from .env file
load_dotenv()

WORKING_DIR = "./radix_docs_embed_3072_o3_mini"
if not os.path.exists(WORKING_DIR): 
    os.mkdir(WORKING_DIR)

embed_model = os.getenv("EMBED_MODEL_LARGE")
llm_model=os.getenv("REASONING_LLM_MODEL")
api_key=os.getenv("API_KEY")
api_version=os.getenv("BASE_LLM_API_VERSION")
api_embed_version=os.getenv("EMBED_API_VERSION")
azure_endpoint=os.getenv("BASE_API_BASE")

azure_embed_endpoint=os.getenv("EMBED_API_BASE")

azure_point=os.getenv("BASE_FOR_LIGHTRAG")


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs
) -> str:
    # Rename max_tokens to max_completion_tokens if present (for newer models than 4o)
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    return await azure_openai_complete_if_cache(
        llm_model,   # deployment *name*
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=azure_point,
        api_version=api_version,
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await azure_openai_embed(
        model=embed_model,  # e.g. text-embed-3-sm
        texts=texts,
        api_key=api_key,
        base_url=azure_point,
        api_version=api_embed_version  # or 4096 depending on the deployment
    )  # or 4096 depending on the deployment

# --- Function from radix_docs_crawler.py to get the sitemap URLs ---
def get_radix_docs_urls():
    """
    Fetches all URLs from the Radix docs site.
    Uses the sitemap (https://radix.equinor.com/sitemap.xml) to get these URLs.
    Returns:
        List[str]: List of URLs
    """            
    sitemap_url = "https://radix.equinor.com/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

# --- New function to fetch and combine markdown content ---
async def fetch_all_radix_docs_contents() -> str:
    """
    Crawls all URLs fetched from the Radix docs sitemap.
    Retrieves the markdown content for each URL and combines them into a single string.
    """
    urls = get_radix_docs_urls()
    if not urls:
        print("No URLs found to crawl.")
        return ""
    
    print(f"Found {len(urls)} URLs to crawl.")
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    markdown_contents = []

    async def fetch_url(url: str):
        try:
            result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
            if result.success:
                print(f"Successfully crawled: {url}")
                # Append the markdown content from this URL
                markdown_contents.append(result.markdown.raw_markdown)
            else:
                print(f"Failed to crawl: {url} - {result.error_message}")
        except Exception as ex:
            print(f"Exception crawling {url}: {ex}")

    # Limit concurrency if needed:
    await asyncio.gather(*[fetch_url(url) for url in urls])
    
    await crawler.close()
    
    # Combine all markdown into a single string separated by a delimiter (for example, a double newline)
    combined_markdown = "\n\n".join(markdown_contents)
    return combined_markdown

# --- RAG Initialization ---
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,  # Adjust based on your model
            max_token_size=8191,  # Adjust based on your model
            func=embedding_func  # Use the OpenAI embedding function
        ),
        llm_model_func=llm_model_func
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def main():
    # Retrieve combined markdown from all Radix docs URLs
    combined_markdown = asyncio.run(fetch_all_radix_docs_contents())
    if not combined_markdown:
        print("No documentation content retrieved.")
        return

    # Initialize RAG instance and insert combined documentation
    rag = asyncio.run(initialize_rag())
    rag.insert(combined_markdown)
    print("Inserted combined documentation into RAG.")

if __name__ == "__main__":
    main()
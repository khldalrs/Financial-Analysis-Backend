# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from openai import OpenAI
# import json
# import yfinance as yf
# import concurrent.futures
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import requests
# import os

# app = FastAPI()

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# PINECONE_INDEX_NAME = "stocks"
# index = Pinecone.Index(PINECONE_INDEX_NAME)
# namespace = "stock-descriptions"

# Pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# hf_embeddings = HuggingFaceEmbeddings()
# vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=hf_embeddings)


# main.py

# from fastapi import FastAPI, HTTPException
# from dotenv import load_dotenv
# import os
# from pinecone import Pinecone, ServerlessSpec
# from langchain.vectorstores import Pinecone as LangChainPineconeVectorStore
# from langchain.embeddings import HuggingFaceEmbeddings

# app = FastAPI()

# load_dotenv()


# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# PINECONE_INDEX_NAME = "stocks"
# namespace = "stock-descriptions"

# # Initialize Pinecone client with required parameters
# pc = Pinecone(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_ENVIRONMENT
# )

# # Specify the index
# index = pc.Index(PINECONE_INDEX_NAME)

# # pc.list_indexes()

# SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
#                         '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

# IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
#                '__pycache__', '.next', '.vscode', 'vendor'}

# # Initialize embeddings with explicit model_name to avoid deprecation warning
# hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Initialize LangChain's Pinecone VectorStore
# vectorstore = LangChainPineconeVectorStore(
#     index=index,
#     embedding=hf_embeddings,
#     text_key="text",
#     namespace=namespace
# )


# def test_query(query: str):
#     results = vectorstore.similarity_search(query, k=5, namespace=namespace)
#     print("Query:", query)
#     print("Number of results:", len(results))
#     for i, result in enumerate(results):
#         print(f"Result {i+1}:")
#         print("Text:", result.page_content)
#         print("Metadata:", result.metadata)
#         print("---")
        
# test_query("What are companies that are related with apple")




from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangChainPineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "stocks"
namespace = "stock-descriptions"


origins = [
    "http://localhost:3000",  # Next.js frontend
    # Add your production frontend URL here, e.g., "https://yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

try:
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
    logger.info("Pinecone client initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize Pinecone client: %s", str(e))
    raise

# Check if the index exists; if not, create it
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    try:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,  # Ensure this matches your embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',    # Adjust based on your cloud provider
                region='us-east-1'  # Adjust based on your region
            )
        )
        logger.info(f"Created Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        logger.error("Failed to create Pinecone index: %s", str(e))
        raise

# Specify the index
try:
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    logger.error("Failed to connect to Pinecone index: %s", str(e))
    raise

# Initialize embeddings with explicit model_name to avoid deprecation warning
try:
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info("HuggingFaceEmbeddings initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize HuggingFaceEmbeddings: %s", str(e))
    raise

# Initialize LangChain's Pinecone VectorStore with text_key
try:
    vectorstore = LangChainPineconeVectorStore(
        index=index,
        embedding=hf_embeddings,
        text_key="text",  # Ensure this matches the key in your documents
        namespace=namespace
    )
    logger.info("LangChain Pinecone VectorStore initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize LangChain Pinecone VectorStore: %s", str(e))
    raise

class SearchRequest(BaseModel):
    query: str
    k: int = 5  # Default number of results

class SearchResult(BaseModel):
    text: str
    metadata: Dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.post("/research", response_model=SearchResponse)
def research(search: SearchRequest):
    if not search.query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")
    try:
        results = vectorstore.similarity_search(search.query, k=search.k)
        json_results = [
            SearchResult(text=r.page_content, metadata=r.metadata) for r in results
        ]
        return SearchResponse(query=search.query, results=json_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Optional: Test query when running the script directly
def test_query(query: str):
    try:
        results = vectorstore.similarity_search(query, k=5)
        logger.info("Query: %s", query)
        logger.info("Number of results: %d", len(results))
        for i, result in enumerate(results):
            logger.info("Result %d:", i+1)
            logger.info("Text: %s", result.page_content)
            logger.info("Metadata: %s", result.metadata)
            logger.info("---")
    except Exception as e:
        logger.error("Error during test query: %s", str(e))







# def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
#     """
#     Returns:
#         np.ndarray: The generated embeddings as a NumPy array.
#     """
#     model = SentenceTransformer(model_name)
#     return model.encode(text)


# def cosine_similarity_between_sentences(sentence1, sentence2):
#     # Get embeddings for both sentences
#     embedding1 = np.array(get_huggingface_embeddings(sentence1))
#     embedding2 = np.array(get_huggingface_embeddings(sentence2))

#     # Reshape embeddings for cosine_similarity function
#     embedding1 = embedding1.reshape(1, -1)
#     embedding2 = embedding2.reshape(1, -1)

#     # Calculate cosine similarity
#     similarity = cosine_similarity(embedding1, embedding2)
#     similarity_score = similarity[0][0]
#     print(f"Cosine similarity between the two sentences: {similarity_score:.4f}")
#     return similarity_score

# def get_stock_info(symbol: str) -> dict:
#     """
#     Retrieves and formats detailed information about a stock from Yahoo Finance.

#     Args:
#         symbol (str): The stock ticker symbol to look up.

#     Returns:
#         dict: A dictionary containing detailed stock information, including ticker, name,
#               business summary, city, state, country, industry, and sector.
#     """
#     data = yf.Ticker(symbol)
#     stock_info = data.info

#     properties = {
#         "Ticker": stock_info.get('symbol', 'Information not available'),
#         'Name': stock_info.get('longName', 'Information not available'),
#         'Business Summary': stock_info.get('longBusinessSummary'),
#         'City': stock_info.get('city', 'Information not available'),
#         'State': stock_info.get('state', 'Information not available'),
#         'Country': stock_info.get('country', 'Information not available'),
#         'Industry': stock_info.get('industry', 'Information not available'),
#         'Sector': stock_info.get('sector', 'Information not available')
#     }

#     return properties


# data = yf.Ticker("NVDA")
# stock_info = data.info
# print(stock_info)


# def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
#     """
#     Returns:
#         np.ndarray: The generated embeddings as a NumPy array.
#     """
#     model = SentenceTransformer(model_name)
#     return model.encode(text)


# def cosine_similarity_between_sentences(sentence1, sentence2):
#     # Get embeddings for both sentences
#     embedding1 = np.array(get_huggingface_embeddings(sentence1))
#     embedding2 = np.array(get_huggingface_embeddings(sentence2))

#     # Reshape embeddings for cosine_similarity function
#     embedding1 = embedding1.reshape(1, -1)
#     embedding2 = embedding2.reshape(1, -1)

#     # Calculate cosine similarity
#     similarity = cosine_similarity(embedding1, embedding2)
#     similarity_score = similarity[0][0]
#     print(f"Cosine similarity between the two sentences: {similarity_score:.4f}")
#     return similarity_score

# aapl_info = get_stock_info('AAPL')
# print(aapl_info)

#Example usage

# aapl_description = aapl_info['Business Summary']

# company_description = "I want to find companies that make smartphones and are headquarted in California"

# similarity = cosine_similarity_between_sentences(aapl_description, company_description)

# def get_company_tickers():
#     # URL to fetch the raw JSON file from GitHub
#     url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"

#     # Making a GET request to the URL
#     response = requests.get(url)

#     # Checking if the request was successful
#     if response.status_code == 200:
#         # Parse the JSON content directly
#         company_tickers = json.loads(response.content.decode('utf-8'))

#         # Optionally save the content to a local file for future use
#         with open("company_tickers.json", "w", encoding="utf-8") as file:
#             json.dump(company_tickers, file, indent=4)

#         print("File downloaded successfully and saved as 'company_tickers.json'")
#         return company_tickers
#     else:
#         print(f"Failed to download file. Status code: {response.status_code}")
#         return None

# company_tickers = get_company_tickers()
# print(len(company_tickers))

#Pinecone

# pinecone_api_key = PINECONE_API_KEY
# os.environ['PINECONE_API_KEY'] = pinecone_api_key

# index_name = "stocks"
# namespace = "stock-descriptions"

# hf_embeddings = HuggingFaceEmbeddings()
# vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)

# for idx, stock in company_tickers.items():
#     stock_ticker = stock['ticker']
#     stock_data = get_stock_info(stock_ticker)
#     stock_description = stock_data['Business Summary']

#     # Convert any None values in stock_data to a default string
#     for key, value in stock_data.items():
#         if value is None:
#             stock_data[key] = "No data available"

#     # Ensure stock_description is a valid string
#     if not stock_description or not isinstance(stock_description, str):
#         stock_description = "No summary available"

#     print(f"Processing stock {idx} / {len(company_tickers)} :", stock_ticker)

#     vectorstore_from_documents = PineconeVectorStore.from_documents(
#         documents=[Document(page_content=stock_description, metadata=stock_data)],
#         embedding=hf_embeddings,
#         index_name=index_name,
#         namespace=namespace
#     )



#     # Initialize tracking lists
# successful_tickers = []
# unsuccessful_tickers = []

# # Load existing successful/unsuccessful tickers
# try:
#     with open('successful_tickers.txt', 'r') as f:
#         successful_tickers = [line.strip() for line in f if line.strip()]
#     print(f"Loaded {len(successful_tickers)} successful tickers")
# except FileNotFoundError:
#     print("No existing successful tickers file found")

# try:
#     with open('unsuccessful_tickers.txt', 'r') as f:
#         unsuccessful_tickers = [line.strip() for line in f if line.strip()]
#     print(f"Loaded {len(unsuccessful_tickers)} unsuccessful tickers")
# except FileNotFoundError:
#     print("No existing unsuccessful tickers file found")

# def process_stock(stock_ticker: str) -> str:
#     # Skip if already processed
#     if stock_ticker in successful_tickers:
#         return f"Already processed {stock_ticker}"

#     try:
#         # Get and store stock data
#         stock_data = get_stock_info(stock_ticker)
#         stock_description = stock_data['Business Summary']

#         # Store stock description in Pinecone
#         vectorstore_from_texts = PineconeVectorStore.from_documents(
#             documents=[Document(page_content=stock_description, metadata=stock_data)],
#             embedding=hf_embeddings,
#             index_name=index_name,
#             namespace=namespace
#         )

#         # Track success
#         with open('successful_tickers.txt', 'a') as f:
#             f.write(f"{stock_ticker}\n")
#         successful_tickers.append(stock_ticker)

#         return f"Processed {stock_ticker} successfully"

#     except Exception as e:
#         # Track failure
#         with open('unsuccessful_tickers.txt', 'a') as f:
#             f.write(f"{stock_ticker}\n")
#         unsuccessful_tickers.append(stock_ticker)

#         return f"ERROR processing {stock_ticker}: {e}"

# def parallel_process_stocks(tickers: list, max_workers: int = 10) -> None:
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_ticker = {
#             executor.submit(process_stock, ticker): ticker
#             for ticker in tickers
#         }

#         for future in concurrent.futures.as_completed(future_to_ticker):
#             ticker = future_to_ticker[future]
#             try:
#                 result = future.result()
#                 print(result)

#                 # Stop on error
#                 if result.startswith("ERROR"):
#                     print(f"Stopping program due to error in {ticker}")
#                     executor.shutdown(wait=False)
#                     raise SystemExit(1)

#             except Exception as exc:
#                 print(f'{ticker} generated an exception: {exc}')
#                 print("Stopping program due to exception")
#                 executor.shutdown(wait=False)
#                 raise SystemExit(1)

# # Prepare your tickers
# tickers_to_process = [company_tickers[num]['ticker'] for num in company_tickers.keys()]

# # Process them
# parallel_process_stocks(tickers_to_process, max_workers=10)

# def test_query(query: str):
#     results = vectorstore.similarity_search(query, k=5)
#     print("Query:", query)
#     print("Number of results:", len(results))
#     for i, result in enumerate(results):
#         print(f"Result {i+1}:")
#         print("Text:", result.page_content)
#         print("Metadata:", result.metadata)
#         print("---")

# test_query("What are companies that are related with tech?")


# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from openai import OpenAI
# import json
# import yfinance as yf
# import concurrent.futures
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import requests
# import os

# app = FastAPI()

# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# PINECONE_INDEX_NAME = "stocks"

# def get_stock_info(symbol: str) -> dict:
#     data = yf.Ticker(symbol)
#     stock_info = data.info
#     properties = {
#         "Ticker": stock_info.get('symbol', 'Information not available'),
#         'Name': stock_info.get('longName', 'Information not available'),
#         'Business Summary': stock_info.get('longBusinessSummary'),
#         'City': stock_info.get('city', 'Information not available'),
#         'State': stock_info.get('state', 'Information not available'),
#         'Country': stock_info.get('country', 'Information not available'),
#         'Industry': stock_info.get('industry', 'Information not available'),
#         'Sector': stock_info.get('sector', 'Information not available')
#     }
#     return properties

# data = yf.Ticker("NVDA")
# stock_info = data.info
# print(stock_info)

# def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
#     model = SentenceTransformer(model_name)
#     return model.encode(text)

# def cosine_similarity_between_sentences(sentence1, sentence2):
#     embedding1 = np.array(get_huggingface_embeddings(sentence1))
#     embedding2 = np.array(get_huggingface_embeddings(sentence2))
#     embedding1 = embedding1.reshape(1, -1)
#     embedding2 = embedding2.reshape(1, -1)
#     similarity = cosine_similarity(embedding1, embedding2)
#     similarity_score = similarity[0][0]
#     print(f"Cosine similarity between the two sentences: {similarity_score:.4f}")
#     return similarity_score

# sentence1 = "I like walking to the park"
# sentence2 = "I like running to the office"
# similarity = cosine_similarity_between_sentences(sentence1, sentence2)

# aapl_info = get_stock_info('AAPL')
# print(aapl_info)

# aapl_description = aapl_info['Business Summary']
# company_description = "I want to find companies that make smartphones and are headquarted in California"
# similarity = cosine_similarity_between_sentences(aapl_description, company_description)

# def get_company_tickers():
#     url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"
#     response = requests.get(url)
#     if response.status_code == 200:
#         company_tickers = json.loads(response.content.decode('utf-8'))
#         with open("company_tickers.json", "w", encoding="utf-8") as file:
#             json.dump(company_tickers, file, indent=4)
#         print("File downloaded successfully and saved as 'company_tickers.json'")
#         return company_tickers
#     else:
#         print(f"Failed to download file. Status code: {response.status_code}")
#         return None

# company_tickers = get_company_tickers()
# print(len(company_tickers))

# pinecone_api_key = PINECONE_API_KEY
# os.environ['PINECONE_API_KEY'] = pinecone_api_key

# index_name = "stocks"
# namespace = "stock-descriptions"

# hf_embeddings = HuggingFaceEmbeddings()
# vectorstore = PineconeVectorStore(index_name=index_name, embedding=hf_embeddings)

# # Initialize tracking lists
# successful_tickers = []
# unsuccessful_tickers = []

# # Load existing successful/unsuccessful tickers before deciding which tickers to process
# try:
#     with open('successful_tickers.txt', 'r') as f:
#         successful_tickers = [line.strip() for line in f if line.strip()]
#     print(f"Loaded {len(successful_tickers)} successful tickers")
# except FileNotFoundError:
#     print("No existing successful tickers file found")

# try:
#     with open('unsuccessful_tickers.txt', 'r') as f:
#         unsuccessful_tickers = [line.strip() for line in f if line.strip()]
#     print(f"Loaded {len(unsuccessful_tickers)} unsuccessful tickers")
# except FileNotFoundError:
#     print("No existing unsuccessful tickers file found")

# # Create a mapping from ticker to index for printing progress
# ticker_to_idx = {}
# for i, num in enumerate(company_tickers.keys()):
#     ticker_val = company_tickers[num]['ticker']
#     ticker_to_idx[ticker_val] = i

# def process_stock(stock_ticker: str) -> str:
#     # Check if already processed
#     if stock_ticker in successful_tickers:
#         return f"Already processed {stock_ticker}"

#     try:
#         stock_data = get_stock_info(stock_ticker)
#         stock_description = stock_data['Business Summary']
#         if not stock_description or not isinstance(stock_description, str):
#             stock_description = "No summary available"

#         # Print progress similar to previous code
#         idx = ticker_to_idx[stock_ticker]
#         print(f"Processing stock {idx} / {len(company_tickers)} : {stock_ticker}")

#         vectorstore_from_texts = PineconeVectorStore.from_documents(
#             documents=[Document(page_content=stock_description, metadata=stock_data)],
#             embedding=hf_embeddings,
#             index_name=index_name,
#             namespace=namespace
#         )

#         # Record success
#         with open('successful_tickers.txt', 'a') as f:
#             f.write(f"{stock_ticker}\n")
#         successful_tickers.append(stock_ticker)

#         return f"Processed {stock_ticker} successfully"

#     except Exception as e:
#         # Record failure
#         with open('unsuccessful_tickers.txt', 'a') as f:
#             f.write(f"{stock_ticker}\n")
#         unsuccessful_tickers.append(stock_ticker)

#         return f"ERROR processing {stock_ticker}: {e}"

# def parallel_process_stocks(tickers: list, max_workers: int = 10) -> None:
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_ticker = {
#             executor.submit(process_stock, ticker): ticker
#             for ticker in tickers
#         }

#         for future in concurrent.futures.as_completed(future_to_ticker):
#             ticker = future_to_ticker[future]
#             try:
#                 result = future.result()
#                 print(result)
#                 if result.startswith("ERROR"):
#                     print(f"Stopping program due to error in {ticker}")
#                     executor.shutdown(wait=False)
#                     raise SystemExit(1)
#             except Exception as exc:
#                 print(f'{ticker} generated an exception: {exc}')
#                 print("Stopping program due to exception")
#                 executor.shutdown(wait=False)
#                 raise SystemExit(1)

# # Build tickers_to_process after loading successful_tickers, so we skip already processed ones
# tickers_to_process = [company_tickers[num]['ticker'] for num in company_tickers.keys() if company_tickers[num]['ticker'] not in successful_tickers]

# parallel_process_stocks(tickers_to_process, max_workers=10)


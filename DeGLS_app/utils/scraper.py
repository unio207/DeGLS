import requests
from bs4 import BeautifulSoup
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import json
import openai

# Set your OpenAI API key if you want optional formatting
openai.api_key = "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"

# List of URLs to process
urls = [
    "https://example.com/gls-management-1",
    "https://example.com/gls-severity-scale"
]

# Function to clean raw text
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove references
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^a-zA-Z0-9.,:%\-() ]', '', text)  # Remove weird symbols
    return text.strip()

# Optional: Auto-format text chunk using GPT
def format_with_gpt(chunk):
    prompt = f"""
You are an expert agronomist. Reformat the following agronomy information into a clear, concise, and self-contained advisory note.

Your goals:
1. Extract any available details about crop type, disease stage, severity levels, environmental conditions, or recommended actions.
2. Present the information in a structured, readable format.
3. If any information is not mentioned, omit that part. Do not make up missing information.
4. Remove redundant or unclear sentences.
5. Keep the factual accuracy intact.

Format your output as follows (omit any section if data is missing):

Crop/Disease Context: [if available]

Severity Diagnosis: [if available]

Recommended Actions: [if available]

Source Info: [if available]

Here is the information:

{chunk}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700
        )
        formatted_text = response['choices'][0]['message']['content'].strip()
        return formatted_text
    except Exception as e:
        print(f"GPT formatting failed: {e}")
        return chunk  # Return original if GPT fails

# Function to process one URL
def process_url(url):
    print(f"Processing: {url}")
    try:
        # Use Newspaper to extract main content
        article = Article(url)
        article.download()
        article.parse()
        raw_text = article.text
        title = article.title
    except Exception as e:
        print(f"Error extracting article: {e}")
        return []
    
    cleaned_text = clean_text(raw_text)
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(cleaned_text)
    
    # Process each chunk
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        formatted = format_with_gpt(chunk)  # Optional GPT formatting
        
        # Metadata inference (simple example, you can improve this)
        metadata = {
            "source": url,
            "title": title,
            "chunk_number": i + 1,
            "topic": "GLS Management",
            "region": "General"
        }
        processed_chunks.append({
            "content": formatted,
            "metadata": metadata
        })
    
    return processed_chunks

# Master list to hold all processed chunks
knowledge_base = []

# Loop through all URLs
for url in urls:
    kb_chunks = process_url(url)
    knowledge_base.extend(kb_chunks)

# Save to JSON
with open("gls_knowledge_base.json", "w") as f:
    json.dump(knowledge_base, f, indent=2)

print(f"Done! Processed {len(knowledge_base)} chunks saved to gls_knowledge_base.json")

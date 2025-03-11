import requests
from bs4 import BeautifulSoup
from googlesearch import search
from transformers import pipeline
import time

# Use the Hugging Face summarizer model (facebook/bart-large-cnn)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to perform Google search and get URLs
def get_search_results(query, num_results=5):
    urls = []
    for url in search(query, num_results=num_results, lang="en"):
        urls.append(url)
    return urls

# Function to scrape the content from the web pages
def scrape_content(url):
    try:
        # Send request to URL and parse the page content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all paragraphs from the page (you can fine-tune this to extract better content)
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Function to summarize the research content
def summarize_research(content):
    # Clean the content (optional step for long content)
    if len(content) > 1000:  # If the content is large, take the first 1000 characters to summarize
        content = content[:1000]
    
    # Summarize using BART
    summary = summarizer(content, max_length=150, min_length=80, do_sample=False, temperature=0.7)
    
    return summary[0]['summary_text']

# Function to perform deep research on a given topic
def perform_research(query):
    # Get the search results (URLs)
    urls = get_search_results(query)
    
    # Collect content from each URL
    full_content = ""
    for url in urls:
        print(f"Scraping content from: {url}")
        page_content = scrape_content(url)
        full_content += page_content
        time.sleep(2)  # Adding a delay to avoid overloading servers
    
    # Summarize the collected content
    print("\nSummarizing the content...")
    summary = summarize_research(full_content)
    
    return summary

# Main function to run the research process
if __name__ == "__main__":
    # Take input from the user for the research query
    research_query = input("Enter the topic you want to research: ")
    
    # Perform the research and summarize it
    research_summary = perform_research(research_query)
    
    # Print the summary
    print("\nResearch Summary:")
    print(research_summary)
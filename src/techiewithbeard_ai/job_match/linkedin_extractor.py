"""
LinkedIn Job Extractor

Simple web scraper to extract job requirements from LinkedIn job postings.
Uses requests + BeautifulSoup for reliable extraction.
"""

from typing import Any

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class LinkedInJobExtraction(BaseModel):
    """Extracted job data from LinkedIn posting"""

    job_title: str
    company: str
    job_description: str
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    location: str | None = None
    employment_type: str | None = None
    seniority_level: str | None = None


def extract_linkedin_job_sync(url: str, config: Any = None) -> dict[str, Any]:
    """
    Extract job requirements from a LinkedIn job posting URL.
    
    Uses simple HTTP requests + BeautifulSoup parsing for reliable extraction.
    No LLM or agents needed - just web scraping.
    
    Args:
        url: LinkedIn job posting URL
        config: Model configuration (optional, not used for simple extraction)
        
    Returns:
        Dict with:
            - success: bool
            - job_text: extracted text content (if successful)
            - error: error message (if failed)
            - url: source URL
    """
    try:
        # Set a proper user agent to avoid being blocked
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract all text content
        text_content = soup.get_text(separator='\n', strip=True)
        
        # Limit to reasonable size (5000 chars)
        text_content = text_content[:5000] if len(text_content) > 5000 else text_content
        
        return {
            "success": True,
            "job_text": text_content,
            "url": url,
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. LinkedIn may be blocking the connection.",
            "url": url,
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Connection error. Please check your internet connection.",
            "url": url,
        }
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else None

        if status_code == 403:
            return {
                "success": False,
                "error": "Access forbidden. LinkedIn may be blocking automated access.",
                "url": url,
            }
        elif status_code == 404:
            return {
                "success": False,
                "error": "Job posting not found. Please check the URL.",
                "url": url,
            }
        else:
            return {
                "success": False,
                "error": f"HTTP Error {status_code or 'unknown'}",
                "url": url,
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Extraction failed: {str(e)}",
            "url": url,
        }

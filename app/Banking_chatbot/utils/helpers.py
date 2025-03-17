import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


def ensure_directory_exists(dir_path: str):
    """Ensure that a directory exists, creating it if necessary"""
    os.makedirs(dir_path, exist_ok=True)


def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    """Format source information for display in the response"""
    if not sources:
        return "No sources used."
    
    formatted = "Sources:\n"
    for i, source in enumerate(sources, 1):
        source_type = source.get("source_type", "unknown")
        
        if source_type == "web":
            # Format web source
            title = source.get("title", "Untitled")
            url = source.get("url", "No URL")
            formatted += f"{i}. {title} - {url}\n"
        else:
            # Format document source
            content_preview = source.get("content_preview", "")
            formatted += f"{i}. Document snippet: {content_preview}\n"
    
    return formatted


def log_event(event_type: str, data: Dict[str, Any], log_file: Optional[str] = None):
    """Log an event to a file for monitoring/debugging"""
    if not log_file:
        log_dir = "logs"
        ensure_directory_exists(log_dir)
        log_file = os.path.join(log_dir, f"{event_type}.log")
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        **data
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(event) + "\n")


def parse_banking_terms(text: str) -> List[str]:
    """
    Extract banking-related terms from text
    Useful for highlighting key terms or generating tags
    """
    banking_terms = [
        "account", "balance", "transfer", "deposit", "withdrawal",
        "mortgage", "loan", "interest", "credit", "debit", "card",
        "savings", "checking", "statement", "fee", "charge", "payment",
        "transaction", "overdraft", "atm", "bank", "finance", "money",
        "investment", "fund", "portfolio", "stock", "bond", "market",
        "rate", "apr", "principal", "term", "maturity", "insurance"
    ]
    
    found_terms = []
    text_lower = text.lower()
    
    for term in banking_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms
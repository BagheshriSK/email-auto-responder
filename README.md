# Enterprise Email Auto-Responder with RAG
A smart email auto-responder system designed for enterprise sales teams that leverages RAG (Retrieval Augmented Generation) to provide contextually relevant responses using custom sales collateral.

## Overview
This system automatically processes incoming sales emails by:
1. Monitoring a Gmail inbox for new messages
2. Categorizing incoming emails using AI
3. Retrieving relevant context from uploaded sales materials
4. Generating concise, contextually appropriate responses
5. Maintaining email thread continuity
6. Managing email status (read/unread)

## Features
- **Email Monitoring**: Continuous monitoring of Gmail inbox for new messages
- **Smart Categorization**: AI-powered categorization of emails into 8 distinct categories
- **RAG Pipeline**: Uses custom sales

## Tech Stack

Core Framework: Python 3.x
LLM Integration: Ollama (Mistral model)
Vector Database: Chroma DB
Email Service: Gmail API

## Key Libraries

LangChain & LangChain Community
Google API Python Client
ChromaDB
PyPDF, docx2txt
Email Validator


## Installation

Clone the repository:

```git clone [your-repository-url]
cd [repository-name]
```

Install required packages:

```
pip install -r requirements.txt
```


Set up Gmail API credentials:

1. Create a Google Cloud Project
2. Enable Gmail API
3. Download credentials and save as:
credentials_primary.json for main email account (account to read emails from)
credentials_secondary.json for secondary email account (send on behalf of account)


## Usage

Add sales collateral:
Place your sales documents in the data/directory
# Supported formats: PDF, DOCX, TXT

Initialize the vector store:
```python fetch_context.py
```

Start the auto-responder:
```python main.py --interval 30  # Checks for new emails every 30 seconds
```


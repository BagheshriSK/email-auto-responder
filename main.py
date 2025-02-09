import os
import base64
import pickle
import re
import time
import argparse
from typing import List, Dict, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Constants remain the same
CHROMA_PATH = "chroma"
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly', 
    'https://www.googleapis.com/auth/gmail.modify'
]

def fetch_new_emails(service, last_processed_time: float = None) -> List[Dict[str, str]]:
    """
    Fetch new unread emails since the last processed time.
    """
    # If no last_processed_time provided, use current time
    if last_processed_time is None:
        last_processed_time = time.time()
        print(f"Initial run - will only process emails after: {time.ctime(last_processed_time)}")
    
    results = service.users().messages().list(
        userId='me', 
        labelIds=['UNREAD'],
        q=f'is:unread after:{int(last_processed_time)}'  # Add time filter to Gmail query
    ).execute()
    messages = results.get('messages', [])
    
    new_emails = []
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        
        timestamp = float(msg.get('internalDate', 0)) / 1000.0  # Convert from milliseconds to seconds
        
        if timestamp <= last_processed_time:
            continue
            
        # Rest of the email processing remains the same
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
        
        parts = payload.get('parts', [])
        body = ''
        if parts:
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    body = base64.urlsafe_b64decode(part.get('body', {}).get('data', '')).decode('utf-8')
                    break
        else:
            body = msg.get('snippet', '')
        
        _, sender_email = parseaddr(sender or '')
        
        user_email = service.users().getProfile(userId='me').execute().get('emailAddress', '')
        user_domain = get_domain(user_email)
        sender_domain = get_domain(sender_email)
        
        if user_domain.lower() == sender_domain.lower():
            continue
        
        new_emails.append({
            'id': message['id'], 
            'thread_id': msg.get('threadId'),
            'body': body, 
            'subject': subject, 
            'sender': sender,
            'sender_email': sender_email,
            'timestamp': timestamp
        })
    
    return new_emails

def extract_first_name(sender: str) -> str:
    """Extract first name from email sender string."""
    name = re.sub(r'<.*>', '', sender).strip()
    name_parts = name.split()
    return name_parts[0] if name_parts else 'there'

def get_domain(email: str) -> str:
    """Extract domain from an email address."""
    _, parsed_email = parseaddr(email)
    domain_match = re.search(r'@(.+)', parsed_email)
    return domain_match.group(1) if domain_match else ''

def authenticate_gmail(account: str) -> Credentials:
    """
    Authenticate with Gmail API for a specific account and return a service object.
    """
    creds = None
    token_file = f'token_{account}.pickle'
    credentials_file = f'credentials_{account}.json'

    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def get_embedding_function():
    """
    Returns the embedding function for Chroma database.
    """
    return OllamaEmbeddings(model="nomic-embed-text")

def search_vector_store(query: str, num_results: int = 3, similarity_threshold: float = 0.7) -> List[Document]:
    """
    Search both vector stores for documents most similar to the query.
    Returns only relevant documents based on a similarity threshold.
    """
    # Connect to both vector stores
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
    # Perform similarity search in both stores
    results = db.similarity_search_with_score(query, k=num_results)
    
    relevant_documents = []
    
    # Process combined results to filter by similarity threshold
    for result in results:
        # Ensure result is a tuple (document, score) or just a document
        if isinstance(result, tuple):
            document, score = result
        else:
            document, score = result, None  # Assuming score is None if not a tuple

        # Add the document if it meets the similarity threshold
        if score is not None and score >= similarity_threshold:
            relevant_documents.append(document)
        
    return relevant_documents

def generate_email_reply(email_content: str, context_documents: List[Document], first_name: str) -> str:
    """
    Generate an extremely brief email reply only if relevant context exists.
    """
    # Check if we have any relevant documents
    if not context_documents:
        print("No relevant context found. Skipping reply.")
        return ""
    
    llm = ChatOllama(model="mistral")
    
    prompt_template = PromptTemplate(
        input_variables=["email", "context", "name"],
        template="""Generate an extremely brief, direct email reply. Maximum 30 words.

Context:
{context}

Incoming Email:
{email}

Guidelines:
- Address recipient by first name: {name}
- Be extremely concise
- Focus on core message
- Use simple, clear language
- No pleasantries or filler words

Reply:"""
    )
    
    # Ensure the context is passed correctly as a string
    context_string = "\n".join([doc.page_content for doc in context_documents])
    
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: context_string)
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke({
        "email": email_content, 
        "context_docs": context_documents,
        "name": first_name
    })
    
    return response

def categorize_email(email_content: str) -> int:
    """
    Categorize email using Ollama's Mistral LLM.
    Returns category number (1-6).
    """
    llm = ChatOllama(model="mistral")
    
    categorization_prompt = PromptTemplate(
    input_variables=["email"],
    template="""You are an expert email categorization assistant. Carefully analyze the following email and categorize it into one of these precise categories:

1. Prospect wants to learn more and has asked specific questions about a product or service (e.g., pricing, features, availability, etc.)
2. Prospect wants to connect but hasn't mentioned a specific time or date (e.g., casual introduction or general interest in connection)
3. Prospect wants to connect and has specified a time or date for a meeting (e.g., scheduling a meeting or call with a proposed time)
4. Prospect finds the offer or information irrelevant but might be interested at a later time (e.g., requesting to be contacted later or expressing disinterest for now but leaving the door open for future communication)
5. Prospect wants to be removed from the mailing list or requests not to be contacted again (e.g., explicitly asking to unsubscribe or cease communication)
6. The email contains feedback, complaints, or responses unrelated to a potential connection or product inquiry (e.g., user feedback, complaints, or general comments)
7. The email is purely informational, no request or inquiry (e.g., sending a newsletter, update, or other unsolicited informational content without a request for action)
8. Others (anything else that does not fall into the categories above, such as spam, greetings, unclear or incomplete messages)

Email Content:
{email}

Guidelines for Categorization:
- Read the entire email carefully.
- Pay attention to keywords or phrases indicating interest, time-specific requests, or disinterest.
- Look for direct requests (e.g., "Please unsubscribe," "I'd like to learn more," "Can we schedule a time to talk?")
- Be precise in your categorization.
- If unsure or the email is ambiguous, choose category 8 ("Others").

Your response should ONLY be the NUMBER of the category. Do not include any additional text.

Categorization:"""
)
    
    chain = (
        categorization_prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        category = int(chain.invoke({"email": email_content}).strip())
        return category if 1 <= category <= 6 else 6
    except ValueError:
        return 6  # Default to Others if we can't parse the response

def send_email_in_thread(service, thread_id: str, to: str, body: str):
    """
    Send a reply in the same email thread, preserving thread context.
    """
    try:
        # Fetch the full thread details
        thread = service.users().threads().get(userId='me', id=thread_id).execute()
        original_message = thread['messages'][-1]  # Get the last message in the thread
        
        # Extract key headers from the original message
        headers = original_message['payload'].get('headers', [])
        
        # Find subject, message-id, and other relevant headers
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
        message_id = next((h['value'] for h in headers if h['name'].lower() == 'message-id'), '')
        
        # Prepare the reply message
        message = MIMEMultipart()
        message['to'] = to
        message['subject'] = f"Re: {subject}" if not subject.startswith('Re:') else subject
        
        # Add thread-related headers
        if message_id:
            message['In-Reply-To'] = message_id
            message['References'] = message_id
        
        # Add message body
        message.attach(MIMEText(body, 'plain'))
        
        # Encode the message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        
        # Send the message in the same thread
        send_message = service.users().messages().send(
            userId='me', 
            body={
                'raw': raw_message, 
                'threadId': thread_id
            }
        ).execute()
        
        print(f"Threaded reply sent successfully. Message Id: {send_message['id']}")
        return send_message
    
    except Exception as error:
        print(f'An error occurred while sending email: {error}')
        return None


def mark_email_processed(service, message_id: str):
    """
    Mark an email as processed by removing the UNREAD label.
    """
    try:
        service.users().messages().modify(
            userId='me', 
            id=message_id, 
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        print(f"Marked message {message_id} as processed.")
    except Exception as error:
        print(f'An error occurred while marking email as processed: {error}')

def process_single_email(
    service,
    email: Dict[str, str],
    context_documents: List[Document]
) -> float:
    """
    Process a single email - categorize it and respond if appropriate.
    Only respond if relevant context is found in the vector store.
    """
    print(f"\nProcessing email from: {email['sender']}")
    
    # First, categorize the email
    category = categorize_email(email['body'])
    print(f"Email category: {category}")
    
    # Only proceed with reply for category 1 (prospect questions)
    if category == 1:
        first_name = extract_first_name(email['sender'])
        
        # Search the vector store for context (with a similarity threshold of 0.3)
        #context_documents = search_vector_store(email['body'], similarity_threshold=0.3)
        
        # Only reply if we have relevant context
        if context_documents:
            # Generate reply using the existing function
            reply = generate_email_reply(
                email['body'],
                context_documents,
                first_name
            )
            
            # Send the reply
            if reply:
                send_email_in_thread(
                    service,
                    thread_id=email['thread_id'],
                    to=email['sender_email'],
                    body=reply
                )
            else:
                print("Generated reply is empty or irrelevant.")
        else:
            print("No relevant context found for this email. Skipping reply.")
    
    # Mark as processed regardless of category
    mark_email_processed(service, email['id'])
    
    return email['timestamp']


def watch_and_auto_reply(polling_interval: int = 30):
    """
    Continuously watch for new emails and auto-reply to appropriate ones.
    """
    service = authenticate_gmail("primary")
    last_processed_time = time.time()  # Start with current time
    print(f"Starting email auto-responder. Will process emails received after: {time.ctime(last_processed_time)}")
    
    try:
        while True:
            new_emails = fetch_new_emails(service, last_processed_time)
            
            if new_emails:
                print(f"Found {len(new_emails)} new emails to process")
                
            for email in new_emails:
                # Search vector store for context
                context_documents = search_vector_store(email['body'], similarity_threshold=0.3)
                
                # Process the email and get its timestamp
                timestamp = process_single_email(service, email, context_documents)
                
                # Update last processed time
                last_processed_time = max(last_processed_time, timestamp)
            
            # Wait before next check
            time.sleep(polling_interval)
    
    except KeyboardInterrupt:
        print("\nEmail auto-responder stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    """Main function to start email auto-responder."""
    parser = argparse.ArgumentParser(description="Email Auto-Responder")
    parser.add_argument("--interval", type=int, default=30, 
                        help="Polling interval in seconds (default: 30)")
    args = parser.parse_args()
    
    watch_and_auto_reply(args.interval)

if __name__ == "__main__":
    main()
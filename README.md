Agent Email Bot
===============

A LangChain-based RAG agent that answers questions from documents and sends emails automatically.

Features
--------

*   **RAG Integration**: Retrieves context from documents using Chroma vector database
    
*   **Smart Agent**: Automatically chooses between query-only, email-only, or combined operations
    
*   **Email Automation**: Sends emails via Gmail SMTP when requested
    
*   **Conversation Memory**: Remembers chat history across sessions
    
*   **Reranking**: Uses BGE reranker to improve document relevance
    
*   **Hallucination Prevention**: Answers only based on retrieved documents
    

Quick Start
-----------

### 1\. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2\. Environment Setup

Create .env file:

```env
OPENAI_API_KEY=your-openai-api-key
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-gmail-app-password
```

### 3\. Populate Database
```bash
python fill_db.py
```
### 4\. Run Agent
```bash
python agent_email.py
```
Usage Examples
--------------

**Query only:**
```
你的問題：請問美國的三大速食連鎖店分別是什麼
答案：美國的三大速食連鎖店分別是麥當勞、溫娣和漢堡王。
```
**Query + Email:**
```
你的問題：請問美而美是什麼，並把答案傳給 test@example.com
答案：美而美是台灣的早餐店連鎖品牌...
✅ 已寄出至 test@example.com
```
**Reference previous answer:**
```
你的問題：把剛剚的答案寄給 user@example.com
答案：✅ 已寄出至 user@example.com
```
How It Works
------------

The agent uses three tools based on request type:

1.  **fetch\_rag\_context** - Query documents only
    
2.  **answer\_and\_email** - Query + send email in one step
    
3.  **send\_email** - Send existing content via email
```
User Input → Agent Analysis → Tool Selection → Action → Response
```
Key Features
------------

### RAG Pipeline

1.  Vector search using BGE-M3 embeddings
    
2.  Rerank results with BGE reranker
    
3.  Generate answers with GPT-4o-mini
    
4.  Ground answers in retrieved documents
    

### Conversation Memory

*   Persistent chat history in chat\_logs/
    
*   Understands references like "previous question"
    
*   Maintains context across sessions
    

### Error Handling

*   Automatic retry on parsing errors
    
*   Fallback to CPU if GPU unavailable
    
*   Graceful handling of missing documents
    

Architecture
------------

*   **Agent**: ReAct pattern with tool selection
    
*   **Vector Store**: Chroma with BGE-M3 embeddings
    
*   **LLM**: OpenAI GPT-4o-mini
    
*   **Memory**: FileChatMessageHistory
    
*   **Reranker**: BAAI BGE reranker
    

Troubleshooting
---------------

**No documents found:**
```bash
python fill_db.py  # Populate database
```
**Email auth error:**
```
Use Gmail App Password, not regular password
```
**OpenAI API error:**
```
Check OPENAI_API_KEY in .env file
```
**Out of scope questions:**
```
System responds: "背景資料中沒有相關資訊"
```
File Structure
--------------
```
├── Agent/
│   ├── agent_email.py       # Main agent
│   ├── fill_db.py          # Database setup
│   └── chat_logs/          # Conversation history
├── chroma_db/              # Vector database
├── data/                   # Source documents
├── .env                    # Environment variables
└── requirements.txt
```
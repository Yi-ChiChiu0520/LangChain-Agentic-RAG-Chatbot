import json
import os
import re
from typing import List
import smtplib
from email.message import EmailMessage

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

# ===== 基本設定 =====
DB_DIR = "chroma_db"  # 向量資料庫儲存目錄
COLLECTION = "my-collection"  # Chroma 資料庫集合名稱
SESSION_ID = "ethan-test-new"  # Changed session ID to start fresh
load_dotenv()  # 載入 .env.example（SENDER_EMAIL / SENDER_PASSWORD）


# --- 工具函式 ---
def flatten_text(text: str) -> str:
    """壓縮多餘空白，避免 reranker 因換行/雜訊影響分數"""
    return re.sub(r"\s+", " ", text).strip()


# ===== 載入交叉編碼器 Reranker（啟動時只載一次）=====
_RERANKER_TOK = None
_RERANKER = None

try:
    _RERANKER_TOK = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    _RERANKER = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

    # Check if CUDA is available and move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _RERANKER = _RERANKER.to(device)
    _RERANKER.eval()
    print(f"✅ Reranker loaded successfully on {device}")
except Exception as e:
    print(f"⚠️ Failed to load reranker model: {e}")
    print("⚠️ Reranking will be disabled, using original retrieval order")


def rerank(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    使用交叉編碼器對檢索到的段落重新排序，回傳前 top_k 段。
    如果模型載入失敗，則回傳原始順序的前 top_k 段。
    """
    if _RERANKER is None or _RERANKER_TOK is None:
        print("[RERANK] Model not available, using original order")
        return docs[:top_k]

    try:
        scored = []
        device = next(_RERANKER.parameters()).device

        for d in docs:
            text = flatten_text(d.page_content)
            inputs = _RERANKER_TOK(query, text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                score = _RERANKER(**inputs).logits.item()
            scored.append((score, d))
            # 除錯：檢視每段分數
            print(f"[RERANK] score={score:.4f} | meta={d.metadata}")

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]
    except Exception as e:
        print(f"[RERANK] Error during reranking: {e}, using original order")
        return docs[:top_k]


# ===== RAG 元件：Embedding、VectorStore、Retriever =====
try:
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"⚠️ Failed to load embedding model: {e}")
    print("⚠️ Using fallback embedding model")
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Fallback embedding model loaded successfully")
    except Exception as e2:
        print(f"❌ Failed to load fallback embedding model: {e2}")
        raise RuntimeError("No embedding model available")

try:
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=DB_DIR,
    )
    print("✅ Vector store initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize vector store: {e}")
    raise RuntimeError("Vector store initialization failed")

# 初始召回 k 筆（交給 reranker 再排序）
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ===== LLM（OpenAI）=====
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize LLM: {e}")
    print("⚠️ Please ensure OPENAI_API_KEY is set in your environment")
    raise RuntimeError("LLM initialization failed")


# ===== 工具（Tools）=====
@tool("fetch_rag_context", return_direct=True)
def fetch_rag_context(question: str) -> str:
    """
    根據問題，從向量庫抓取相關段落並重排序，回傳完整的答案。
    """
    print(f"Searching for: {question}")
    initial_docs = retriever.invoke(question)

    if not initial_docs:
        return "根據檢索結果，查無相關資料。"

    top_docs = rerank(question, initial_docs, top_k=5)
    context = "\n\n---\n\n".join([d.page_content for d in top_docs])
    print(f"Context found: {context[:200]}...")

    # Use LLM to generate answer based on context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一個專業的問答助手。根據提供的背景資料回答用戶問題。\n\n"
         "規則：\n"
         "1. 只根據背景資料回答，不要添加背景資料中沒有的資訊\n"
         "2. 如果背景資料中沒有答案，請明確說明\n"
         "3. 回答要簡潔明確\n"
         "4. 保持客觀中性\n\n"
         "背景資料：\n{context}"),
        ("human", "問題：{question}")
    ])

    qa_chain = qa_prompt | llm
    response = qa_chain.invoke({
        "context": context,
        "question": question
    })
    return response.content.strip()


@tool("send_email", return_direct=True)
def send_email_tool(input_data: str) -> str:
    """
    寄送 Email。輸入應為 JSON 字串格式：
    {"to": "收件人email", "subject": "主旨", "body": "內容"}
    使用 Gmail（需在 .env.example 設定 SENDER_EMAIL / SENDER_PASSWORD App Password）
    """
    sender = os.getenv("SENDER_EMAIL")
    pwd = (os.getenv("SENDER_PASSWORD") or "").replace(" ", "")
    if not sender or not pwd:
        return "⚠️ 缺少 SENDER_EMAIL / SENDER_PASSWORD 環境變數。"

    try:
        # Parse the JSON input
        if isinstance(input_data, str):
            data = json.loads(input_data)
        else:
            data = input_data

        to = data["to"]
        subject = data["subject"]
        body = data["body"]

        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, pwd)
            s.send_message(msg)

        print(f"[EMAIL]\nTO: {to}\nSUBJECT: {subject}\nBODY:\n{body}\n")
        return f"✅ 已寄出至 {to} [[EMAIL_SENT]]"  # Added marker for system prompt
    except json.JSONDecodeError as e:
        return f"⚠️ JSON 解析錯誤：{e}"
    except KeyError as e:
        return f"⚠️ 缺少必要欄位：{e}"
    except Exception as e:
        return f"⚠️ 寄信失敗：{e}"


@tool("answer_and_email", return_direct=True)
def answer_and_email_tool(input_data: str) -> str:
    """
    回答問題並同時寄送答案到指定的 email。輸入應為 JSON 字串格式：
    {"question": "要回答的問題", "email": "收件人email", "subject": "郵件主旨（可選）"}
    """
    try:
        if isinstance(input_data, str):
            data = json.loads(input_data)
        else:
            data = input_data

        question = data["question"]
        email = data["email"]
        subject = data.get("subject", "回答您的問題")

        # First, get the answer using RAG
        print(f"Searching for: {question}")
        initial_docs = retriever.invoke(question)

        if not initial_docs:
            answer = "根據檢索結果，查無相關資料。"
        else:
            top_docs = rerank(question, initial_docs, top_k=5)
            context = "\n\n---\n\n".join([d.page_content for d in top_docs])
            print(f"Context found: {context[:200]}...")

            # Use LLM to generate answer based on context
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "你是一個專業的問答助手。根據提供的背景資料回答用戶問題。\n\n"
                 "規則：\n"
                 "1. 只根據背景資料回答，不要添加背景資料中沒有的資訊\n"
                 "2. 如果背景資料中沒有答案，請明確說明\n"
                 "3. 回答要簡潔明確\n"
                 "4. 保持客觀中性\n\n"
                 "背景資料：\n{context}"),
                ("human", "問題：{question}")
            ])

            qa_chain = qa_prompt | llm
            response = qa_chain.invoke({
                "context": context,
                "question": question
            })
            answer = response.content.strip()

        # Send email
        sender = os.getenv("SENDER_EMAIL")
        pwd = (os.getenv("SENDER_PASSWORD") or "").replace(" ", "")
        if not sender or not pwd:
            return f"答案：{answer}\n⚠️ 缺少 SENDER_EMAIL / SENDER_PASSWORD 環境變數，無法寄信。"

        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = email
        msg["Subject"] = subject
        msg.set_content(answer)

        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, pwd)
            s.send_message(msg)

        print(f"[EMAIL]\nTO: {email}\nSUBJECT: {subject}\nBODY:\n{answer}\n")
        return f"答案：{answer}\n✅ 已寄出至 {email}"

    except json.JSONDecodeError as e:
        return f"⚠️ JSON 解析錯誤：{e}"
    except KeyError as e:
        return f"⚠️ 缺少必要欄位：{e}"
    except Exception as e:
        return f"⚠️ 處理請求時發生錯誤：{e}"


tools = [fetch_rag_context, send_email_tool, answer_and_email_tool]

# ===== ReAct Agent（由提示決定是否使用工具）=====
react_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是助理 Adam。你必須嚴格按照 ReAct 格式回應。\n\n"
     "重要：仔細分析用戶的請求！\n\n"
     "工具使用規則：\n"
     "- 如果用戶問問題並明確要求寄信（包含「寄給」、「傳給」、「寄信給」、「發送給」、「email to」等關鍵字），使用 answer_and_email 工具\n"
     "- 如果用戶只是問問題（沒有寄信要求），使用 fetch_rag_context 工具\n"
     "- 如果用戶只是要求寄信已知內容（不需要查詢），使用 send_email 工具\n\n"
     "注意：\n"
     "1. 如果問題需要從資料庫查詢答案並寄信，使用 answer_and_email\n"
     "2. 如果問題需要從資料庫查詢答案但不寄信，使用 fetch_rag_context\n"
     "3. 如果只是寄信已知內容，使用 send_email\n\n"
     "你必須嚴格按照以下格式回應：\n"
     "Thought: [仔細分析用戶請求：1)問了什麼問題？2)是否要求寄信？3)是否需要查詢資料？4)應該使用哪個工具？]\n"
     "Action: [工具名稱]\n"
     "Action Input: [輸入內容]\n\n"
     "可用工具：{tools}\n"
     "工具名稱：{tool_names}\n\n"
     "範例 1（只問問題，需要查詢）：\n"
     "用戶：什麼是美而美？\n"
     "你的回應：\n"
     "Thought: 用戶問了關於美而美的問題，需要查詢資料，但沒有要求寄信，我需要使用 fetch_rag_context\n"
     "Action: fetch_rag_context\n"
     "Action Input: \"什麼是美而美？\"\n\n"
     "範例 2（問問題+寄信，需要查詢）：\n"
     "用戶：什麼是美而美？並寄給 test@example.com\n"
     "你的回應：\n"
     "Thought: 用戶問了問題並要求寄信，需要查詢資料，我需要使用 answer_and_email 工具\n"
     "Action: answer_and_email\n"
     "Action Input: {{\"question\": \"什麼是美而美？\", \"email\": \"test@example.com\", \"subject\": \"回答您的問題\"}}\n\n"
     "範例 3（只寄信已知內容）：\n"
     "用戶：寄送\"Hello World\"給 test@example.com\n"
     "你的回應：\n"
     "Thought: 用戶要求寄信已知內容，不需要查詢資料，我需要使用 send_email 工具\n"
     "Action: send_email\n"
     "Action Input: {{\"to\": \"test@example.com\", \"subject\": \"信件內容\", \"body\": \"Hello World\"}}"
     ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}"),
])

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="錯誤：你同時輸出了 Action 和 Final Answer。請只輸出一個步驟：要嘛是 'Thought/Action/Action Input'，要嘛是 'Final Answer'，不能同時輸出。請重新開始當前步驟。",
    max_iterations=2,  # Reduced since all tools return directly
    early_stopping_method="force",  # Changed back to force since generate is not supported
    return_intermediate_steps=False
)


# ===== 對話記憶（持久化到檔案）=====
def get_session_history(session_id: str):
    os.makedirs("chat_logs", exist_ok=True)
    return FileChatMessageHistory(f"chat_logs/{session_id}.json")


# 把 AgentExecutor 包裝進記憶機制
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# ===== 對外回答介面（單一路徑：交給 Agent 決策是否用工具）=====
def answer(user_input: str) -> str:
    """
    交給 ReAct Agent：它會先 fetch_rag_context，再回答；
    若使用者要求寄信，就會自動呼叫 send_email。
    """
    try:
        result = agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        # AgentExecutor 的輸出是 dict，最終答案在 "output"
        return result.get("output", str(result))
    except Exception as e:
        print(f"Error in answer function: {e}")
        return f"抱歉，處理您的請求時發生錯誤：{e}"


# ===== CLI =====
if __name__ == "__main__":
    print("🔎 啟動聊天（Agent 模式），輸入 'exit' 離開")
    while True:
        q = input("\n你的問題：").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("\n--- 答案 ---")
        print(answer(q))
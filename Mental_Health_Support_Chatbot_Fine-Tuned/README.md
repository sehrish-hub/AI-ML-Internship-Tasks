# 🧠 Hybrid Mental Health Support Chatbot (Agent + Fine-Tuned Model + RAG + Streamlit UI)

**Gemini-2.5-Flash Agent(OpenAI Agents SDK) + Fine-Tuned DistilGPT2 + Retrieval Augmented Generation (RAG) + Streamlit UI**

---

# 📌 Project Overview

This project is a production-level **hybrid AI mental health support chatbot** that combines:

* ✅ Fine-Tuned DistilGPT2 → for empathetic emotional responses
* ✅ Gemini-2.5-Flash Agent → for intelligence, reasoning, safety and orchestration
* ✅ RAG (FAISS Vector Database) → knowledge retrieval
* ✅ OpenAI Agents SDK → for agent orchestration
* ✅ HuggingFace Transformers → for fine-tuning and inference
* ✅ Sentence Transformers → embeddings generation
* ✅ Streamlit → browser-based user interface for chat

This hybrid architecture provides:

* Emotionally intelligent responses
* Context-aware support using RAG
* Safe and controlled outputs
* Agent-based orchestration
* Professional production architecture
* Browser-based chat UI

---

# 🧠 Architecture Overview

```
User Input
   ↓
Gemini Agent (Reasoning + Safety + Tool Selection)
   ↓
rag_tool (Retrieves knowledge from FAISS vector DB)
   ↓
empathetic_response Tool
   ↓
Fine-Tuned DistilGPT2 Model
   ↓
Final Supportive Response
   ↓
Streamlit Browser UI
```

---

# 🧠 Hybrid AI Architecture Components

**1️⃣ Gemini Agent (Brain)**

Responsible for:
* Understanding user intent
* Selecting appropriate tools
* Ensuring safety
* Orchestrating the workflow

Model used:
* gemini-2.5-flash

**2️⃣ Fine-Tuned DistilGPT2 (Emotional Intelligence Layer)**

Responsible for:

* Generating empathetic responses
* Emotional understanding
* Human-like supportive replies

Fine-tuned on:
* EmpatheticDialogues Dataset (Facebook AI)

**3️⃣ RAG System (Knowledge Layer)**

Responsible for:
* Retrieving relevant mental health knowledge
* Providing context-aware responses
* Improving response accuracy

Uses:
* Sentence Transformers
* FAISS Vector Database

**4️⃣ Streamlit UI (User Interaction Layer)**

Responsible for:
* Browser-based chat interface
* Maintaining chat session
* Displaying conversation history
* Sending user input to agent for response

--- 

# ⚙️ Technologies Used

| Technology                  | Purpose                         |
| --------------------------- | ------------------------------- |
| Python                      | Programming language            |
| OpenAI Agents SDK           | Agent orchestration             |
| Gemini-2.5-Flash            | Reasoning model                 |
| HuggingFace Transformers    | Fine-tuning and inference       |
| DistilGPT2                  | Base language model             |
| Sentence Transformers       | Embeddings generation           |
| FAISS                       | Vector database                 |
| PyTorch                     | Deep learning backend           |
| dotenv                      | Environment variable management |
| NumPy                       | Vector processing               |
| Streamlit                   | Browser UI for chatbot          |

---

# 📂 Project Structure

```
Mental_Health_Support_Chatbot_Fine-Tuned/
│
├── mental_health_model/ # Fine-tuned model folder
│
├── rag_system.py
│
├── hybrid_chatbot.py # Main hybrid chatbot (Gemini + fine-tuned model)
│
├── app.py # Streamlit browser UI
│
├── train_model.py # Fine-tuning script
│
├── .env # API keys
│
├── requirements.txt # Dependencies
│
└── README.md # Documentation
```

---

# 🔑 Environment Setup

Create `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

---

# 📦 Installation

## Step 1: Create virtual environment

```
python -m venv .venv
```

Activate:

Windows:

```
.venv\Scripts\activate
```

Mac/Linux:

```
source .venv/bin/activate
```

---

## Step 2: Install dependencies

```
pip install torch transformers datasets python-dotenv sentence-transformers faiss-cpu openai-agents
```

---

# 🤖 Step 1: Fine-Tune Model DistilGPT2

Run:

```
python train_model.py
```

This will:

* Load EmpatheticDialogues dataset
* Train DistilGPT2
* Save model in:

```
./mental_health_model
```

---

# 🧠 Step 2: Run Hybrid Chatbot with RAG

Run:

```
python hybrid_chatbot.py
```

Output:

```
Mental Health Chatbot Ready ✅

You: I feel stressed about exams

Bot: I understand how overwhelming exams can feel...
```

---

# 🌐 Step 3: Run Streamlit Browser Chat UI

Run:
streamlit run app.py
This will:

* Open a browser window
* Display chat interface
* Maintain session-based conversation
* Send user messages to agent (Gemini + Fine-Tuned Model + RAG)
* Display bot responses in chat bubbles
---

# 🧠 How It Works

1. Knowledge converted into embeddings (Sentence Transformers)
2. Embeddings stored in FAISS vector DB
3. User input retrieved context using `rag_tool`
4. Context + user input sent to fine-tuned DistilGPT2 via `empathetic_response`
5. Gemini Agent orchestrates tool usage
6. Streamlit UI displays conversation in browser


## Step 1: Convert knowledge into embeddings(Sentence Transformers)

* Example knowledge:
Stress is a normal response to challenging situations.
Deep breathing can help calm anxiety.
Sleep is important for emotional well-being.

* Converted into vectors using:
SentenceTransformer("all-MiniLM-L6-v2")

## Step 2: Store embeddings in FAISS vector DB
FAISS stores vectors for fast similarity search.

## Step 3: Retrieve relevant context
Example:
User Input:
I feel stressed
Retrieved context:
Stress is a normal response to challenging situations.
Exercise helps reduce stress hormones.

## Step 4: Send context to Fine-Tuned Model
Final prompt:
Context:
Stress is a normal response...

User: I feel stressed
Bot:

---

## 🧩 Agent Tool Architecture
Mental Health Agent
│
├── Model:
│     Gemini-2.5-Flash
│
├── Tools:
│     ├── rag_tool
│     └── empathetic_response
│
└── Runner:
      Executes agent

---
## 🛠 Tools Explained

# 🛠 Tool Function

# Tool 1: rag_tool
Responsible for:
Retrieving knowledge from vector DB
rag_tool(user_input)

# Tool 2: empathetic_response
Responsible for:
Generating empathetic responses
Using fine-tuned DistilGPT2
empathetic_response(user_input)

```
@function_tool
def empathetic_response(user_input: str) -> str:
```

This function:

* Takes user input
* Sends to fine-tuned model
* Returns empathetic response

---

# 🎯 Features

✅ Hybrid AI Architecture  
✅ Fine-Tuned Emotional Model  
✅ Gemini AI Agent Reasoning  
✅ Tool-Based Agent Design  
✅ RAG Knowledge Retrieval  
✅ Streamlit Browser Chat UI  
✅ Safe & Empathetic Responses  
✅ Modular & Production-Ready

---

# 🧪 Example Interaction

Input:

```
I feel anxious about my future
```

Output:

```
It’s completely understandable to feel anxious about the future.
You’re not alone in feeling this way, and it’s okay to take things one step at a time.
```

---


# 🔐 Safety Features
System ensures:
* No medical diagnosis
* No harmful advice
* Supportive emotional responses only
* Agent-controlled tool usage
* Safe reasoning via Gemini

---

# 🚀 Production-Level Features

✅ Hybrid AI architecture
✅ Agent orchestration
✅ Fine-tuned emotional model
✅ RAG knowledge retrieval
✅ Tool-based design
✅ Modular architecture
✅ Context-aware responses
✅ Industry-level architecture
---


# 👩‍💻 Author

Developed by: Sehrish Shafiq

AI Engineer | Agentic AI Developer

# Specialization:

AI Agents
LLM Engineering
RAG Systems
Agentic AI Architecture

---

# 📜 License

MIT License

---

# ⭐ Summary

This project demonstrates a production-level hybrid AI system combining:

* Agent orchestration
* Fine-tuned LLM
* RAG system
* Tool-based architecture
* Emotional intelligence
* Streamlit Browser Chat UI  

This is a real AI Engineer portfolio-level project.

---

Notes:

torch → Deep learning backend for Transformers & DistilGPT2

transformers → Load & fine-tune LLMs

datasets → HuggingFace datasets (EmpatheticDialogues)

sentence-transformers → Embeddings for RAG

faiss-cpu → Vector database for similarity search

numpy → Required by FAISS

python-dotenv → Load .env with API keys

openai-agents → Gemini agent orchestration

accelerate → Optional, speeds up training & inference










🧠 Architecture Diagram (conceptual)
User
 │
 ▼
Gemini Agent (brain)
 │
 ├── decides tool needed
 ▼
empathetic_response tool call
 │
 ▼
Fine-tuned DistilGPT2 (empathetic model)
 │
 ▼
Response returned to Gemini (Empathetic response generate)
 │
 ▼
Gemini improves response
 │
 ▼
Final response → User












Complete Workflow Samajhiye (Step-by-Step)
Step 1: Load dataset
        ↓
Step 2: Format into chatbot format
        ↓
Step 3: Convert text → tokens
        ↓
Step 4: Load DistilGPT2 model
        ↓
Step 5: Train model on dataset
        ↓
Step 6: Save trained model
        ↓
Step 7: Load trained model
        ↓
Step 8: Generate empathetic response
Yeh code exactly kya kar raha hai (Simple words)

Yeh code:

• Emotional conversations dataset load karta hai
• DistilGPT2 ko empathetic chatbot banne ke liye train karta hai
• Trained model save karta hai
• Aur phir test karta hai

Final Result

Aapka model ban gaya:

./mental_health_model

Yeh same model aap hybrid agent me tool ke tarah use kar rahe hain.

Industry Architecture me isko kya kehte hain

Yeh hai:

Fine-Tuned Domain-Specific LLM

Aur jab Gemini ke sath use karein:

Hybrid Multi-Model Agent System






















Execution Flow Summary (Step-by-Step)
Documents
   ↓
SentenceTransformer
   ↓
Embeddings (vectors)
   ↓
FAISS index
   ↓
Store vectors

User query:

"I feel anxiety"
   ↓
Convert to embedding
   ↓
FAISS similarity search
   ↓
Find most similar documents
   ↓
Return relevant context
Real Example
context = retrieve_context("I feel stressed")
print(context)

Output:

Stress is a normal response to challenging situations.
Exercise helps reduce stress hormones.
RAG Architecture me iska role
User Question
   ↓
retrieve_context()
   ↓
Relevant knowledge
   ↓
LLM (Gemini / GPT)
   ↓
Final Answer
Ye system kya provide karta hai

This is:

Vector database

Semantic search engine

Knowledge retrieval system

RAG memory layer

Professional use in your Mental Health Chatbot

Flow:

User: I feel anxious
   ↓
retrieve_context()
   ↓
Relevant mental health info
   ↓
Gemini Agent
   ↓
Empathetic + informed response
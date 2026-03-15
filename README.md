# 🤖 MVP RAG: National AI Strategy Advisor
**Author:** Mouhamech Nabil  
**Project:** Fast MVP in Data Science Homework

## 🎯 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer complex questions regarding the "National Strategy for the Development of Artificial Intelligence in Russia until 2030." The goal was to create a stable, factually accurate advisor that mitigates hallucinations by grounding responses strictly in the provided legal document[cite: 4, 1305, 1306].

## 🏗️ System Architecture
The application follows a modular RAG pipeline built with **LangChain**:

1.  **Ingestion & Chunking**: The Strategy PDF is processed using `PyPDFLoader` and split into 2000-character segments with a 200-character overlap to preserve semantic context between sections[cite: 1012, 1283].
2.  **Vector Store**: Text fragments are converted into high-dimensional vectors and stored in a **FAISS** index for rapid semantic retrieval[cite: 1013, 1201].
3.  **Retrieval**: The system uses a similarity search to find the top 5 most relevant document fragments for any given query[cite: 1280, 1281].
4.  **Augmentation & Generation**: The retrieved context is injected into a specialized **System Prompt** and passed to the LLM to generate a final response[cite: 1306, 1437].



## 🛠️ Technical Strategy: Cost & Performance Optimization
A key architectural decision was using **NVIDIA Embeddings + Groq LLM**.
* **NVIDIA Embeddings (`llama-nemotron-embed-1b-v2`)**: Used to create a high-quality vector space. NVIDIA's NIM API provides advanced embedding capabilities that allow for precise retrieval of technical legal terms[cite: 1187, 1257].
* **Groq LLM**: Groq’s LPU (Language Processing Unit) architecture allows for exceptionally fast inference, reducing latency for real-time user interaction[cite: 1438, 1476].
* **Cost Minimization**: Using NVIDIA and Groq through their respective APIs often provides a more cost-effective alternative to standard proprietary models while maintaining high reasoning capabilities. This combination allows for a "Fast MVP" that doesn't sacrifice accuracy for budget[cite: 994, 995, 1334].

## 📊 Result Analysis
I conducted three comparative runs using different model combinations. **Run 1 (NVIDIA + Groq)** was selected for the final submission:

| Metric | Run 1 (NVIDIA + Groq) | Run 3 (ChatGPT) |
| :--- | :--- | :--- |
| **Correction Logic** | **Success**: Identified the "12 billion" error and corrected it to **60 billion**. | **Fail**: Hallucinated an answer based on the incorrect number. |
| **Conciseness** | High: Used structured bullet points. | Low: Included excessive "water" and long tables. |
| **Groundedness** | Strictly followed the PDF context. | Relied too much on internal training data. |

**Key Finding:** The system prompt's **Correction Logic** was essential. It ensured that if a question contained a factual error (like Question 3 regarding target volumes), the model pointed out the mistake and provided the correct data from the document.

## 🚀 How to Run
1.  **Environment**: Install dependencies: `pip install langchain langchain-community langchain-openai faiss-cpu pypdf pandas openpyxl`.
2.  **API Keys**: Add your keys (`NVIDIA_API_KEY`, `GROQ_API_KEY`, etc.) to a `.env` file in the project root[cite: 1387, 1388].
3.  **Execution**: Run the main script:
    ```bash
    python rag_system.py
    ```
4.  **Output**: The system will generate a professionally formatted Excel file named `test_set_Mouhamech_Nabil.xlsx`.

---

**Submission Files Included:**
* `rag_system.py`: [The source code](https://github.com/NabilM5/RAG_MVP/blob/main/rag_system.py).
* `test_set_Mouhamech_Nabil.xlsx`: The final generated answers.
* `README.md`: This documentation.
* **Screencast Link**: [\[Google Drive Click Here\]](https://drive.google.com/drive/folders/1j3sOFy1OAshjkzoSZOEbz6dK01LBmPd2?usp=sharing).
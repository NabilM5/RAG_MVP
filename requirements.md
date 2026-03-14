# 📋 Project Requirements: National AI Strategy RAG MVP

## 1. Functional Requirements
* **Contextual Accuracy**: The system must answer questions strictly based on the "National Strategy for the Development of Artificial Intelligence in Russia until 2030".
* **Hallucination Mitigation**: The model is required to identify and point out factual errors in user queries (specifically the 12 billion vs. 60 billion ruble target discrepancy).
* **Output Format**: All responses must be saved to an Excel file named `test_set_Mouhamech_Nabil.xlsx` with a dedicated `answer` column.
* **Conciseness**: Answers must be direct and brief, avoiding unnecessary "filler" text to improve relevancy scores.

## 2. Technical Requirements
* **Framework**: Built using **LangChain** for modular RAG pipeline management.
* **Embedding Model**: Must use `text-embedding-3-large` to ensure compatibility with the instructor's RAGAS evaluation baseline.
* **LLM Generator**: Configured with `temperature=0` to ensure deterministic and fact-based responses.
* **Vector Store**: Implementation of **FAISS** for efficient semantic similarity search.
* **Resilience**: The code must include retry logic with exponential backoff to handle API Rate Limits (HTTP 429).

## 3. Data Constraints
* **Closed-Domain Knowledge**: No external documents or general internet knowledge may be added to the vector index.
* **Chunking Strategy**: Documents must be split using a recursive strategy (e.g., 2000 characters) with sufficient overlap (200 characters) to maintain semantic integrity.

## 4. Evaluation Metrics (RAGAS)
The system is optimized to perform across five key metrics:
1. **Answer Relevancy**: Alignment of the response to the core intent of the question.
2. **Answer Correctness**: Factual truth compared to the "Ground Truth" in the Strategy.
3. **Answer Similarity**: Semantic closeness to the reference answers.
4. **Clarity**: Logical flow and readability of the formulation.
5. **Safety**: Absence of harmful or unethical content.
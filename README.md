
---

## 🛍️ Amazon FAQ Chatbot (RAG + Groq + FAISS)

**An intelligent chatbot that answers Amazon-style customer queries** using a **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **FAISS**, and **Groq LLaMA 3.1 8B Instant**.
It retrieves the most relevant FAQ context and generates accurate, human-like responses.

---

### 🧠 Project Overview

This project demonstrates how RAG (Retrieval-Augmented Generation) can be used to build a **domain-specific Q&A chatbot**.
It retrieves relevant passages from stored Amazon FAQs and combines them with a large language model to provide faithful, context-aware answers.

The system was also **evaluated manually** using multiple quality metrics like *faithfulness*, *answer relevancy*, *context precision*, and *response time*.

---

### ⚙️ Key Features

✅ Retrieval-Augmented Generation pipeline
✅ Local FAISS vector database for fast search
✅ Embedding using `sentence-transformers/all-MiniLM-L6-v2`
✅ LLM inference via **Groq LLaMA 3.1 8B Instant**
✅ Interactive **Streamlit chatbot interface**
✅ Evaluation with custom metric computations (`manual_evaluation_results.csv`)

---

### 📁 Project Structure

```
Amazon_Rag/
│
├── data/                           # FAQ dataset (Amazon-style Q&A)
├── vectorstore/                    # FAISS vector database
│
├── amazon_faq_rag.ipynb            # Full Jupyter notebook (RAG + Evaluation)
├── manual_evaluation_results.csv   # Evaluation results (faithfulness, relevancy, etc.)
├── app.py                          # Streamlit chatbot app
│
├── requirements.txt                # Python dependencies
├── .env                            # Contains GROQ_API_KEY
└── README.md                       # Project documentation
```

---

### 🧩 Tech Stack

| Component  | Library / Model                                                  |
| ---------- | ---------------------------------------------------------------- |
| LLM        | Groq `llama-3.1-8b-instant`                                      |
| Embedding  | `sentence-transformers/all-MiniLM-L6-v2`                         |
| Retriever  | FAISS                                                            |
| Framework  | LangChain                                                        |
| Interface  | Streamlit                                                        |
| Evaluation | Custom Python metrics (faithfulness, relevancy, precision, etc.) |

---

### 🚀 How to Run

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/Amazon-FAQ-RAG.git
cd Amazon-FAQ-RAG
```

#### 2️⃣ Set up your environment

```bash
python -m venv venv
venv\Scripts\activate       # (Windows)
pip install -r requirements.txt
```

#### 3️⃣ Add API key

Create a `.env` file in the root directory with:

```
GROQ_API_KEY=your_groq_api_key
```

#### 4️⃣ Run the Streamlit chatbot

```bash
streamlit run app.py
```

Visit → `http://localhost:8501`
and start chatting with your **Amazon FAQ Assistant** 🛒

---

### 📊 Evaluation Summary

You can find the evaluation notebook in
`amazon_faq_rag.ipynb`, which generates `manual_evaluation_results.csv`.

Metrics evaluated include:

| Metric                | Description                                     |
| --------------------- | ----------------------------------------------- |
| **Faithfulness**      | How truthful the answer is to retrieved context |
| **Answer Relevancy**  | Alignment between generated answer and query    |
| **Context Precision** | Quality of retrieved documents                  |
| **Response Time**     | Model’s average response speed                  |
| **Answer Length**     | Length distribution for answer generation       |

Visualizations such as bar charts, histograms, and scatter plots were created to analyze these results.

---

### 💬 Example Query

```
Q: How do I return a defective product?
A: To return a defective product, visit the "Your Orders" section, select the item, and choose "Return or Replace". Amazon will generate a return label for you.
```

---

### 🧾 Future Improvements

* Add RAGAS-based automatic evaluation
* Integrate hybrid retrieval (BM25 + embeddings)
* Deploy using **Streamlit Cloud / HuggingFace Spaces**
* Enhance conversational memory for multi-turn chat

---

### 🧑‍💻 Author

**Gokulakrishnan T**
PhD in Mathematics | Aspiring AI Engineer
📬 *Focused on practical ML, RAG systems, and real-world AI applications.*

---



# Q-A-AURORA
## Endpoint ASK URL
```
https://aurora-test-1.onrender.com/api/ask
```
## Project Structure
```
│
├── app/
│ ├── init.py
│ ├── main.py           # FastAPI entrypoint
│ ├── api/
│ │
│ │ └── ask.py          # Core API endpoint logic
│ ├── connectors.py     # External connectors & env setup
│ 
├── EDA.py              # Getting insights from Databawse
├── requirements.txt    # Dependencies 
├── Procfile            # Gunicorn startup command
├── runtime.txt 
├── README.md 
└── .env / Render Variables # API keys and config (not committed)
```
## Overview

This repository implements a question-answering API built using FastAPI and LangChain.  
It retrieves, interprets, and responds to natural-language user queries based on member messages, using a hybrid of semantic retrieval and LLM reasoning.

---

## Approaches Considered

| Method | Description | Advantages | Disadvantages |
|--------|--------------|-------------|----------------|
| 1. Semantic Search Only | Retrieves top-k similar messages using sentence-transformer embeddings and FAISS. | - Fast and lightweight<br>- No LLM cost<br>- High recall for known queries | - Cannot generalize<br>- Misses unseen question patterns<br>- No reasoning or rephrasing |
| 2. Semantic + Regex Rules | Combines embedding similarity with keyword and regex-based entity extraction (e.g., dates, names). | - Adds structure & precision<br>- Detects dates, numbers, and preferences | - Brittle rules<br>- Hard to scale across languages<br>- Still lacks contextual understanding |
| 3. LLM + Top-k Context (Used Here) | Uses a language model (OpenAI / OpenRouter) with top 3–4 retrieved messages as context for each question. | - Balances accuracy and interpretability<br>- Can generalize unseen phrasing<br>- Explains reasoning in natural text | - Slightly slower than pure retrieval<br>- Requires API key and inference cost |
| 4. LLM + Chunking + Rules (Advanced) | Extends (3) by chunking long texts and combining with rule-based reasoning and structured metadata. | - Handles large, multi-turn histories<br>- High factual precision<br>- More scalable to enterprise setups | - High compute and cost<br>- Complex orchestration<br>- Requires caching and monitoring |

---

## Method Used in This Implementation

Approach 3: LLM + Top-k Message Context

- The API retrieves the 3–4 most semantically similar messages using FAISS and SentenceTransformers.  
- The retrieved context is passed to a Large Language Model (via LangChain’s `LLMChain`) to generate a coherent and context-aware answer.  
- This method optimally balances performance, interpretability, and deployment simplicity ideal for lightweight, serverless deployments like Render’s free tier.

  ---

## Data Insights

The message dataset analyzed (`messages_pretty.json`, total: 3349 messages) revealed rich user behavior and data quality findings.

### Dataset Summary
- Total Messages: 3349  
- Unique Users: 10  
- Date Range: Nov 2024 – Nov 2025  
- Average Messages per User: ~10  

### Key Observations
| Category | Finding | Example / Impact |
|-----------|----------|------------------|
| Encoding Errors | 6% messages with UTF-8 issues (`â€™` → `'`) | Affects text embeddings if not cleaned |
| Mixed Date Formats | Mentions like “this Friday”, “November 15”, “first week of December” | Requires date normalization for consistent parsing |
| User Activity Imbalance | 3× difference between most (Sophia Al-Farsi, 16 msgs) and least active users (Amina Van Den Berg, 5 msgs) | Bias in personalized response quality |
| Cultural Name Diversity | Names include non-ASCII (e.g., “Hans Müller”, “Layla Kawaguchi”) | Suggests multilingual dataset potential |
| Sensitive Data | Occasional phone numbers/invoice data (e.g., “555-349-7841”) | Must be masked before embedding for privacy |
| Incomplete Messages | Some truncated entries (“I finally”, “Thank you”) | Should be filtered as non-informative context |
| Temporal Clustering | Users send bursts of related queries over short spans | Useful for building session-based retrieval windows |

### Topic Distribution
| Topic | Percentage of Messages |
|--------|------------------------|
| Travel / Bookings | 29% |
| Customer Service | 18% |
| Account Updates | 15% |
| Restaurant Reservations | 13% |
| Other | 25% |

These insights guide preprocessing, retrieval weighting, and contextual modeling improvements.

---
## Exploratory Data Analysis (EDA)

You can view the full source code for the EDA here:  
[EDA Analysis Script](./eda_analysis.py)

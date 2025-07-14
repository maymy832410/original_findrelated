---
title: Academic Article Search Engine
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "4.25.0"
app_file: app.py
pinned: true
---

# 🧠 Academic Article Search Engine

This is an AI-powered web app for searching and analyzing scholarly articles published by **Mesopotamian Academic Press** and **Peninsula Publishing Press**. It uses keyword extraction, semantic ranking, and citation filtering to streamline literature discovery.

---

## 🚀 Key Features

- 🔍 **AI-Based Keyword Extraction**: Uses KeyBERT + SentenceTransformer to extract meaningful keyphrases.
- 🧠 **Semantic Search Engine**: Uses `all-MiniLM-L6-v2` to rank articles by relevance to the input query.
- 🏷️ **Filter by Publisher**: Supports Mesopotamian Academic Press and Peninsula Publishing Press.
- 📊 **Citation Filtering**: Filter results by Crossref citation count (slider).
- 🧾 **Article Cache**: Stores results locally for fast repeated access. Smart cache appending without duplication.
- 📦 **Update Metadata**: Admins can fetch new metadata using a secure password (`update@press`).
- 📄 **Clean Output Format**: Results are displayed in a scroll-free, mobile-optimized markdown format.
- 🔄 **Real-Time Update Progress**: See how many articles are fetched in real-time.

---

## 💡 How It Works

1. Input a **main topic** (keyword) or **title/abstract**
2. System extracts key phrases using KeyBERT and builds a semantic query
3. Cached metadata is loaded and filtered based on:
   - Publisher selection
   - Citation limit
   - Semantic match using cosine similarity
4. Results are ranked and displayed with citation stats and similarity score
5. Admins can update cache via password-protected button without resetting old data

---

## 🧪 Stack & Libraries

- **UI**: [`gradio`](https://gradio.app/)
- **ML**: `sentence-transformers`, `keybert`, `scikit-learn`
- **Data**: `pandas`, `requests`
- **API Sources**: [Crossref Metadata API](https://api.crossref.org/)

---

## 🛠️ Requirements

To run locally:

```bash
pip install -r requirements.txt
python app.py

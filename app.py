# app.py

import gradio as gr
import pandas as pd
import requests
import time
import re
import functools
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# ------------------ CONFIG -------------------
SCOPUS_API_KEY = "4c82c88cb16a62f87b0b770c06d6a917"
PUBLISHERS = {
    "Mesopotamian Academic Press": "37356",
    "Peninsula Publishing Press": "51231"
}
CACHE_FILE = "cached_articles.pkl"
CACHE_DURATION_SECONDS = 60 * 60 * 24 * 7
model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# ------------------ CACHE -------------------
article_cache = {}
last_cache_time = 0

def load_cache_from_file():
    global article_cache, last_cache_time
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)
            article_cache.update(cache_data["data"])
            last_cache_time = cache_data["timestamp"]

def save_cache_to_file():
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"data": article_cache, "timestamp": last_cache_time}, f)

# ------------------ FETCHING -------------------
def fetch_articles_from_crossref(publisher_id, max_results=500, progress_cb=None):
    base_url = "https://api.crossref.org/works"
    cursor = "*"
    fetched = 0
    articles = []

    while fetched < max_results:
        params = {
            "filter": f"member:{publisher_id},type:journal-article",
            "rows": 100,
            "cursor": cursor,
            "mailto": "mohanad@peninsula-press.ae"
        }
        headers = {
            "User-Agent": "SmartArticleBot/1.0 (mailto:mohanad@peninsula-press.ae)"
        }

        try:
            res = requests.get(base_url, params=params, headers=headers, timeout=10)
            if res.status_code == 429:
                time.sleep(10)
                continue
            if res.status_code != 200:
                break
        except requests.RequestException as e:
            print(f"Error during fetch: {e}")
            time.sleep(5)
            continue

        data = res.json().get("message", {})
        items = data.get("items", [])
        cursor = data.get("next-cursor", None)

        for item in items:
            title = item.get("title", [""])[0]
            abstract = re.sub("<[^<]+?>", "", item.get("abstract", "")).strip()
            keywords = " ".join(item.get("subject", []))
            content = f"{title} {abstract} {keywords}".lower()
            authors = ", ".join(
                [f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])]
            ) if "author" in item else "N/A"
            date_parts = item.get("published-print", item.get("published-online", {})).get("date-parts", [[None]])
            published = "-".join([str(x) for x in date_parts[0]]) if date_parts else "N/A"

            doi = item.get("DOI", "")
            doi_link = f"https://doi.org/{doi}" if doi else ""

            articles.append({
                "Title": title,
                "Authors": authors,
                "DOI": doi_link,
                "Crossref Citations": item.get("is-referenced-by-count", 0),
                "Content": content
            })

            fetched += 1
            if progress_cb:
                progress_cb(f"Fetching from {publisher_id}... Total fetched: {fetched}")
            if fetched >= max_results:
                break

        if not cursor or len(items) < 100:
            break
        time.sleep(0.5)

    return articles

def extract_core_keywords(text, top_n=5):
    try:
        raw_keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            diversity=0.5,
            top_n=15
        )
        phrases = [kw[0] for kw in raw_keywords]
        embeddings = model.encode(phrases)
        final_keywords = []
        used = set()
        for i, phrase in enumerate(phrases):
            if i in used:
                continue
            final_keywords.append(phrase)
            for j in range(i + 1, len(phrases)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > 0.75:
                    used.add(j)
            if len(final_keywords) >= top_n:
                break
        return final_keywords
    except Exception as e:
        print(f"Keyword extraction error: {e}")
        return []

def load_cached_articles():
    global last_cache_time
    now = time.time()
    if article_cache and (now - last_cache_time) < CACHE_DURATION_SECONDS:
        return article_cache["df"]
    all_articles = []
    for name, pub_id in PUBLISHERS.items():
        results = fetch_articles_from_crossref(pub_id, 500)
        for r in results:
            r["Publisher"] = name
        all_articles.extend(results)
    df = pd.DataFrame(all_articles)
    article_cache["df"] = df
    last_cache_time = now
    save_cache_to_file()
    return df

def compute_similarity(df, query):
    query_embed = model.encode(query, convert_to_tensor=True)
    contents = df["Content"].tolist()
    doc_embeds = model.encode(contents, convert_to_tensor=True)
    sims = cosine_similarity([query_embed.cpu().numpy()], doc_embeds.cpu().numpy())[0]
    df["Similarity (%)"] = (sims * 100).round().astype(int)
    return df

def format_results_markdown(df):
    markdown_blocks = []
    for i, row in df.iterrows():
        block = f"""### \U0001F4C4 Article {i + 1}
**Title**: {row['Title']}  
**Authors**: {row['Authors']}  
**DOI**: [{row['DOI']}]({row['DOI']})  
**Crossref Citations**: {row['Crossref Citations']}  
**Similarity**: {row.get('Similarity (%)', 'N/A')}%

---
"""
        markdown_blocks.append(block)
    return "\n".join(markdown_blocks)

def search_articles(query, title_query, publishers, max_citations):
    query = query.strip()
    title_query = title_query.strip()
    df = load_cached_articles()
    total_count = len(df)

    if publishers:
        df = df[df["Publisher"].isin(publishers)]

    extracted_keywords = extract_core_keywords(title_query) if title_query else []
    all_keywords = [query] if query else []
    all_keywords += extracted_keywords

    if not all_keywords:
        return "", "⚠️ Could not extract meaningful keywords from the input.", ""

    pattern = "|".join(map(re.escape, all_keywords))
    try:
        df = df[df["Content"].str.contains(pattern, na=False, case=False, regex=True)]
    except Exception as e:
        return "", f"❌ Error while filtering: {str(e)}", ""

    if df.empty:
        return "", f"⚠️ No results found matching: {', '.join(all_keywords)}", f"🔎 Extracted Keywords: {', '.join(extracted_keywords)}"

    df = compute_similarity(df, " ".join(all_keywords))
    df = df[df["Crossref Citations"] <= max_citations]
    df = df.sort_values(by=["Similarity (%)", "Crossref Citations"], ascending=[False, False])
    markdown_result = format_results_markdown(df)

    return f"🔎 Extracted Keywords: {', '.join(extracted_keywords)}", f"✅ Found {len(df)} / {total_count} publications.", markdown_result

def manual_crossref_update(password):
    if password.strip() != "update@press":
        return "❌ Incorrect password."

    global last_cache_time
    if "df" not in article_cache:
        load_cache_from_file()

    existing_df = article_cache.get("df", pd.DataFrame())
    existing_dois = set(existing_df["DOI"].tolist()) if not existing_df.empty else set()
    all_articles = []

    def progress_log(msg):
        update_status.value = msg

    for name, pub_id in PUBLISHERS.items():
        new_articles = []
        results = fetch_articles_from_crossref(pub_id, 500, progress_cb=progress_log)
        for r in results:
            r["Publisher"] = name
            if r["DOI"] not in existing_dois:
                new_articles.append(r)
        all_articles.extend(new_articles)
        progress_log(f"{name}: {len(new_articles)} new articles added.")

    if all_articles:
        new_df = pd.DataFrame(all_articles)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["DOI"])
    else:
        combined_df = existing_df

    article_cache["df"] = combined_df
    last_cache_time = time.time()
    save_cache_to_file()

    return "✅ Cache updated. {} new articles added.".format(len(all_articles))

# Gradio UI
load_cache_from_file()
with gr.Blocks() as demo:
    gr.Markdown("# \U0001F50D AI-Powered Article Search")
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Main Topic (Query)")
            title_input = gr.Textbox(label="Title or Abstract")
            publisher_filter = gr.CheckboxGroup(choices=list(PUBLISHERS.keys()), label="Select Publisher(s)")
            citation_slider = gr.Slider(0, 500, step=1, label="Max Crossref Citations", value=100)
            search_btn = gr.Button("Search")
            keywords_box = gr.Markdown(visible=True)
            count_box = gr.Markdown(visible=True)
        with gr.Column():
            update_pw = gr.Textbox(type="text", label="Admin Password")
            update_btn = gr.Button("🔄 Update Cache")
            update_status = gr.Textbox(label="Progress", interactive=False, value="Idle")

    results_box = gr.Markdown(visible=True)

    search_btn.click(fn=search_articles, inputs=[query_input, title_input, publisher_filter, citation_slider], outputs=[keywords_box, count_box, results_box])
    update_btn.click(fn=manual_crossref_update, inputs=[update_pw], outputs=[update_status])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))


import pandas as pd
import json
import re
import time
from py2neo import Graph
from openai import OpenAI
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# === CONFIGURATION ===
CSV_PATH = "Financial.csv"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "sito142007"  # ← change if your Neo4j uses a different password

LLM_API_KEY = "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"
LLM_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = ("qwen/qwen3-235b-a22b-2507")


# === 1. DATA PREPROCESSING ===
def load_and_clean_csv(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    # Keep only rows with valid Content and Title
    df = df.dropna(subset=['Content', 'Title'])
    df = df[df['Content'].str.len() > 20]
    print(f"✓ Loaded {len(df)} valid news items.")
    return df


def preprocess_text(texts):
    clean_texts = []
    for text in texts:
        # Remove trailing timestamps like "2023-03-10T19:18:00\""
        text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*$', '', text)
        text = re.sub(r'[^\w\s\.\,\$\%\&\-\(\)]', ' ', text)
        sentences = sent_tokenize(text.strip())
        clean_texts.extend([s.strip() for s in sentences if len(s.strip()) > 10])
    return clean_texts


# === 2. ENTITY & RELATION EXTRACTION (Few-shot) ===
def extract_triplets(sentences):
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    triplets = []

    examples = '''
Example 1:
Text: "Apple Inc. acquired Beats Electronics in 2014."
Output: {"triplets": [{"subject": "Apple Inc.", "subject_type": "Company", "relation": "ACQUIRED", "object": "Beats Electronics", "object_type": "Company", "year": 2014}]}

Example 2:
Text: "Tesla bought SolarCity in 2016."
Output: {"triplets": [{"subject": "Tesla", "subject_type": "Company", "relation": "ACQUIRED", "object": "SolarCity", "object_type": "Company", "year": 2016}]}
'''

    for i, sent in enumerate(sentences[:100]):
        print(f" -> Extracting from sentence {i + 1}/{min(100, len(sentences))}")
        prompt = f"""
You are a financial event extractor. Extract structured triplets from the text.
Only output valid JSON with key "triplets". Use these relation types: ACQUIRED, INVESTED_IN, FOUNDED, ANNOUNCED_EARNINGS, HIT_HIGH, HIT_LOW.
Include "year" if mentioned. If no valid event, output {{"triplets": []}}.

{examples}

Text: "{sent}"
Output:
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                triplets.extend(data.get("triplets", []))
            time.sleep(0.5)
        except Exception as e:
            print(f"Error on sentence: {e}")
    return triplets


# === 3. CLEANING ===
def clean_triplets(triplets):
    cleaned = []
    valid_relations = {"ACQUIRED", "INVESTED_IN", "FOUNDED", "ANNOUNCED_EARNINGS", "HIT_HIGH", "HIT_LOW"}
    for t in triplets:
        rel = t.get("relation", "").upper()
        if rel not in valid_relations:
            continue
        if not t.get("subject") or not t.get("object"):
            continue
        cleaned.append({
            "subject": t["subject"],
            "subject_type": t.get("subject_type", "Company"),
            "relation": rel,
            "object": t["object"],
            "object_type": t.get("object_type", "Company"),
            "year": t.get("year")
        })
    print(f"✓ Cleaned to {len(cleaned)} triplets.")
    return cleaned


# === 4. LOAD TO NEO4J ===
def load_to_neo4j(triplets):
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE")
    graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")

    for t in triplets:
        sub_label = "Person" if t["subject_type"] == "Person" else "Company"
        obj_label = "Person" if t["object_type"] == "Person" else "Company"

        rel_props = ""
        params = {"sub": t["subject"], "obj": t["object"]}
        if t.get("year"):
            rel_props = "{year: $year}"
            params["year"] = t["year"]

        query = f"""
        MERGE (s:{sub_label} {{name: $sub}})
        MERGE (o:{obj_label} {{name: $obj}})
        MERGE (s)-[r:{t['relation']} {rel_props}]->(o)
        """
        graph.run(query, **params)
    print("✓ Graph loaded into Neo4j.")


# === 5. NATURAL LANGUAGE TO CYPHER ===
def nl_to_cypher(question):
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    schema = """
Node labels: Company(name), Person(name)
Relations: ACQUIRED(year), INVESTED_IN(year), FOUNDED(year), ANNOUNCED_EARNINGS, HIT_HIGH, HIT_LOW
"""
    prompt = f"""
Convert this question to a Cypher query using the schema. Output ONLY the Cypher.

Schema:
{schema}

Question: {question}
Cypher:
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


# === MAIN ===
if __name__ == "__main__":
    start_time = time.time()
    print("Starting Knowledge Graph Pipeline...\n")
    df = load_and_clean_csv(CSV_PATH)
    texts = (df["Title"] + ". " + df["Content"]).astype(str).tolist()

    sentences = preprocess_text(texts)
    print(f"Preprocessing done: {len(sentences)} sentences.\n")

    triplets = extract_triplets(sentences)
    print(f"\nRaw triplets extracted: {len(triplets)}\n")
    clean_triplets_list = clean_triplets(triplets)

    if clean_triplets_list:
        load_to_neo4j(clean_triplets_list)
    else:
        print("No valid triplets to load.")
    print("\nDemo: Natural Language to Cypher")
    question = "Which companies hit a new low recently?"
    cypher = nl_to_cypher(question)
    print(f"Q: {question}")
    print(f"Cypher: {cypher}")

    print(f"\nPipeline finished in {time.time() - start_time:.1f} seconds.")
%%writefile src/ingest.py
import os
import shutil
import pandas as pd
import json
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Folders
os.makedirs("src", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 2. Copy Data
try:
    if os.path.exists("/kaggle/input/foodalyse/nutrition_facts_final.csv"):
        shutil.copy("/kaggle/input/foodalyse/nutrition_facts_final.csv", "data/nutrition_facts_final.csv")
        shutil.copy("/kaggle/input/foodalyse/healthy_swaps.json", "data/healthy_swaps.json")
        print("📂 Data Copied.")
except:
    print("⚠️ Check input paths if files are missing.")

# 3. Build Vector DB
print("🔄 Indexing Data...")

if os.path.exists("data/nutrition_facts_final.csv"):
    df = pd.read_csv("data/nutrition_facts_final.csv")
    
    # Load swaps safely
    swaps = {}
    if os.path.exists("data/healthy_swaps.json"):
        with open("data/healthy_swaps.json", "r") as f:
            swaps = json.load(f)

    documents = []
    for _, row in df.iterrows():
        food = row['food']
        tips = ", ".join(swaps.get(food, ["No specific tips"]))
        content = f"Food: {food}\nStats: {row['calories_per_portion']} kcal, {row['protein_g_per_100g']}g protein.\nSwaps: {tips}"
        documents.append(Document(page_content=content, metadata={"food": food}))

    print("🧠 Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("faiss_index")
    print("✅ Database Built Successfully!")
else:
    print("❌ Error: Data files missing.")

if __name__ == "__main__":
    pass
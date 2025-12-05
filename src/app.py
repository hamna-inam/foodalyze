import os
import torch
import gc
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing DeepSeek RAG System...")
    
    # 1. Load Vector DB
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        resources["vector_db"] = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector DB Loaded.")
    except Exception as e:
        print(f"❌ Vector DB Error: {e}")
        resources["vector_db"] = None

    # 2. Load DeepSeek (Strict Memory Management)
    print("🧠 Loading deepseek-ai/deepseek-llm-7b-chat...")
    try:
        # Force garbage collection before load
        gc.collect()
        torch.cuda.empty_cache()

        model_id = "deepseek-ai/deepseek-llm-7b-chat"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Load directly to GPU 0
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map={"": 0}, 
            trust_remote_code=True 
        )
        
        resources["tokenizer"] = tokenizer
        resources["model"] = model
        print("✅ DeepSeek Loaded Successfully!")
        
    except Exception as e:
        print(f"❌ Error Loading DeepSeek: {e}")
        resources["model"] = None

    yield
    # Cleanup
    resources.clear()
    gc.collect()
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class Query(BaseModel):
    text: str

@app.post("/ask")
def ask(q: Query):
    vector_db = resources.get("vector_db")
    model = resources.get("model")
    tokenizer = resources.get("tokenizer")

    # 1. Retrieve
    context = "No context found."
    if vector_db:
        docs = vector_db.similarity_search(q.text, k=3)
        context = "\n".join([d.page_content for d in docs])

    # 2. Generate
    if model:
        try:
            messages = [
                {"role": "user", "content": f"You are a nutritionist. Context: {context}\n\nQuestion: {q.text}"}
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            outputs = model.generate(
                **inputs, 
                max_new_tokens=250,
                do_sample=True,
                temperature=0.7
            )
            
            answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        except Exception as e:
            answer = f"**[Generation Error]** {e}"
    else:
        answer = "**[Mock Answer]** DeepSeek failed to load. Context found:\n" + context

    return {"answer": answer, "context": context}

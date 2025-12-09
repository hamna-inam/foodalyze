import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer

# Import from the file we just created in the same folder
from experiments_strategies import zero_shot, few_shot, chain_of_thought

# --- Configuration ---
MODEL_ID = "deepseek-ai/deepseek-llm-7b-chat"
OUTPUT_FILE = "experiments/results_with_metrics.csv"

# --- 1. Load Model (4-bit for Kaggle) ---
print("🧠 Loading Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)

# --- 2. Setup Evaluation ---
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
strategies = {
    "Zero-Shot": zero_shot,
    "Few-Shot": few_shot,
    "Chain-of-Thought": chain_of_thought,
}

# Load Eval Data
with open("data/eval.jsonl", "r") as f:
    test_set = [json.loads(line) for line in f]

results = []

print(f"🧪 Running Experiments on {len(test_set)} food items...")

# --- 3. Execution Loop ---
for item in test_set:
    food = item["food"]
    truth = item["ground_truth"]

    for strat_name, strat_func in strategies.items():
        print(f"   👉 Testing {strat_name} on '{food}'...")

        # Generate
        prompt = strat_func(food)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=150, do_sample=True, temperature=0.6
        )
        prediction = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

        # Calculate ROUGE (Quantitative)
        scores = scorer.score(truth, prediction)

        # Log Data
        results.append(
            {
                "Food": food,
                "Strategy": strat_name,
                "Ground Truth": truth,
                "AI Prediction": prediction,
                "ROUGE-L": round(scores["rougeL"].fmeasure, 3),
                "Human_Score_1_to_5": "",
            }
        )

# --- 4. Save Results ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Raw results saved to {OUTPUT_FILE}")

# --- 5. Generate Visualization ---
print("📊 Generating Comparison Chart...")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
chart = sns.barplot(data=df, x="Strategy", y="ROUGE-L", palette="viridis")
plt.title("Prompt Strategy Comparison (ROUGE-L Score)")
plt.ylim(0, 1.0)
plt.savefig("experiments/strategy_comparison.png")
print("✅ Chart saved to experiments/strategy_comparison.png")

# --- 6. Summary ---
print("\n🏆 Average ROUGE-L Scores:")
print(df.groupby("Strategy")["ROUGE-L"].mean())

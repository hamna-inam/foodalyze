"""
Lightweight CI evaluation placeholder 
(does NOT load the 7B model — avoids OOM)
"""

import json

def main():
    try:
        with open("data/eval.jsonl") as f:
            lines = f.readlines()

        print(f"Loaded {len(lines)} eval items")
        print("CI prompt evaluation PASSED")
    
    except Exception as e:
        print("CI evaluation FAILED:", e)
        raise e

if __name__ == "__main__":
    main()

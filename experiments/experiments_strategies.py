
def zero_shot(food):
    return f"User: Is {food} healthy? Answer in 2 sentences.\nAssistant:"

def few_shot(food):
    return f"""User: Is palak_paneer healthy?
Assistant: Yes, it is rich in iron and protein, but can be high in fat if too much cream is used. Swap cream for yogurt.

User: Is gulab_jamun healthy?
Assistant: No, it is deep-fried milk solids soaked in sugar syrup, making it very high in calories. Eat sparingly.

User: Is {food} healthy? Answer in 2 sentences.
Assistant:"""

def chain_of_thought(food):
    return f"""System: You are a nutritionist. Follow these steps to answer:
1. Identify the main ingredients of {food}.
2. Analyze the cooking method (fried, steamed, etc.).
3. Conclude if it is healthy or not.

User: Analyze {food}.
Assistant:"""

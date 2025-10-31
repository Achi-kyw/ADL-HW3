from typing import List
import re
    
def get_inference_system_prompt() -> str:
    """get system prompt for generation"""
    prompt = "You are a helpful question-answering assistant. "+\
        "Your primary goal is to answer the user's question accurately and concisely, "+\
        "using *only* the information present in the context passages provided below. "+\
        "Do not use any external knowledge or make assumptions beyond the text. "+\
        "Please make a genuine effort to find the answer within the provided context. Synthesize information across passages if needed. "+\
        "However, it is crucial that your answer is fully supported by the text. "+\
        "If, after careful consideration, you determine that the provided context passages definitively do not contain the information needed to answer the question, "+\
        "and *only* in that situation, respond with the single word : CANNOTANSWER. "+\
        "Otherwise, provide the best possible answer derived solely from the context. "+\
        "Think step-by-step."
    return prompt

def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
    """Create the user prompt for generation given a query and a list of context passages."""
    formatted_context = ""
    if not context_list:
        formatted_context = "No context provided."
    else:
        for i, context in enumerate(context_list, 1):
            formatted_context += f"[Context {i}]: {context.strip()}\n\n"
    prompt = f"""
Here is the user's question:
{query}

---
Here are the relevant context passages I found:
{formatted_context}
---

Based *ONLY* on the context passages provided above, please answer the user's question.
"""
    return prompt

def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text."""
    marker = "<|im_start|>assistant\n"
    pos = pred_ans.rfind(marker)
    
    if pos != -1:
        parsed_ans = pred_ans[pos + len(marker):].strip()
    else:
        marker_simple = "assistant\n"
        pos_simple = pred_ans.rfind(marker_simple)
        if pos_simple != -1:
            parsed_ans = pred_ans[pos_simple + len(marker_simple):].strip()
        else:
            parsed_ans = pred_ans.strip()

    if parsed_ans.endswith("<|im_end|>"):
        parsed_ans = parsed_ans[:-len("<|im_end|>")].strip()
    parsed_ans = re.sub(r'<think>.*?</think>\s*', '', parsed_ans, flags=re.DOTALL)
    return parsed_ans
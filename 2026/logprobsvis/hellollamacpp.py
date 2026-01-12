
from llama_cpp import Llama

model_path = "WRITE PATH TO MODEL!!.gguf"

# Initialize the model
# n_ctx is the context window (adjust based on your RAM/GPU VRAM)
# n_gpu_layers set to -1 will offload all layers to GPU if you have CUDA/Metal configured
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1,
    logits_all=True,
)

# Simple completion
output = llm(
    "Question: Why is 2 + 2 = 4? Answer:",
    max_tokens=64,
    stop=["\n"],
    echo=True
)

print(output["choices"][0]["text"])

import html

def save_probs_to_html(chat_response, filename="probs.html"):
    choices = chat_response["choices"][0]
    if "logprobs" not in choices or choices["logprobs"] is None:
        print("No logprobs found in chat_response")
        return

    token_logprobs = choices["logprobs"]["content"]
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
    body { font-family: sans-serif; line-height: 1.6; padding: 100px; background-color: #f4f4f9; }
    .token-container { display: flex; flex-wrap: wrap; gap: 2px; }
    .token {
        position: relative;
        display: inline-block;
        padding: 2px 4px;
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 3px;
        cursor: help;
    }
    .token:hover { background-color: #eef; }
    .tooltip {
        visibility: hidden;
        width: 250px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 10;
        top: 100%;
        left: 50%;
        margin-left: -125px;
        margin-top: 5px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9em;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .token:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    .tooltip table { width: 100%; border-collapse: collapse; }
    .tooltip th, .tooltip td { padding: 4px; border-bottom: 1px solid #555; }
    .tooltip th { text-align: left; }
    .tooltip .prob-bar {
        height: 5px;
        background-color: #4CAF50;
        display: block;
        margin-top: 2px;
    }
</style>
</head>
<body>
    <h3>Generated Text with Token Probabilities</h3>
    <div class="token-container">
"""

    def get_color(prob):
        # prob is in percentage (0-100)
        # 100% - Black (0, 0, 0)
        # 75% - Blue (0, 0, 255)
        # 50% - Green (0, 128, 0)
        # 25% and below - Red (255, 0, 0)
        
        if prob >= 75:
            # Interpolate between Black (100) and Blue (75)
            # t = 0 at 100%, t = 1 at 75%
            t = (100 - prob) / 25
            r = 0
            g = 0
            b = int(255 * t)
        elif prob >= 50:
            # Interpolate between Blue (75) and Green (50)
            # t = 0 at 75%, t = 1 at 50%
            t = (75 - prob) / 25
            r = 0
            g = int(128 * t)
            b = int(255 * (1 - t))
        elif prob >= 25:
            # Interpolate between Green (50) and Red (25)
            # t = 0 at 50%, t = 1 at 25%
            t = (50 - prob) / 25
            r = int(255 * t)
            g = int(128 * (1 - t))
            b = 0
        else:
            # 25% and below is Red
            r = 255
            g = 0
            b = 0
        
        return f"rgb({r}, {g}, {b})"

    for entry in token_logprobs:
        token_text = entry["token"]
        # The primary token's probability
        import math
        primary_prob = math.exp(entry["logprob"]) * 100
        token_color = get_color(primary_prob)

        # Escape HTML characters and handle spaces/newlines
        display_text = html.escape(token_text).replace(" ", "&nbsp;").replace("\n", "<br>")
        if not display_text.strip() and "&nbsp;" not in display_text and "<br>" not in display_text:
             display_text = "&nbsp;" # Fallback for empty strings if any

        tooltip_html = "<table><tr><th>Token</th><th>Prob</th></tr>"
        for top in entry["top_logprobs"]:
            prob = math.exp(top['logprob']) * 100
            if prob < 0.01:
                continue
            t_text = html.escape(top['token']).replace(" ", "&nbsp;").replace("\n", "↵")
            tooltip_html += f"<tr><td>{t_text}</td><td>{prob:.2f}%<span class='prob-bar' style='width: {prob}%'></span></td></tr>"
        tooltip_html += "</table>"

        html_content += f"""
        <div class="token" style="color: {token_color};">
            {display_text}
            <div class="tooltip">{tooltip_html}</div>
        </div>"""

    html_content += """
    </div>
</body>
</html>
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved probabilities to {filename}")

# Chat completion style (similar to your LM Studio logic)
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer in Russian."},
    {"role": "user", "content": "Оцени по шкале от 0 до 10, насколько хорош С++?"}
]

chat_response = llm.create_chat_completion(
    messages=messages,
    temperature=0.0,
    logprobs=True,
    top_logprobs=5  # Requests top 5 tokens per position
)

# print(chat_response["choices"][0]["message"]["content"])

# Extract logprobs from the response
choices = chat_response["choices"][0]
if "logprobs" in choices and choices["logprobs"] is not None:
    token_logprobs = choices["logprobs"]["content"]

    for i, entry in enumerate(token_logprobs):
        print(f"\nPosition {i + 1}:")
        # top_logprobs contains a list of dicts: [{'token': '...', 'logprob': ...}, ...]
        for top in entry["top_logprobs"]:
            print(f"  Token: {top['token']:10} | Logprob: {top['logprob']:.4f}")
else:
    print("Logprobs not returned. Ensure the model supports them.")

print("\nFull Content:")
print(chat_response["choices"][0]["message"]["content"])

save_probs_to_html(chat_response)

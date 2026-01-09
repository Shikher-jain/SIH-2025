from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser", framework="pt")

def refine_report(text):
    prompt = f"paraphrase: {text} </s>"
    out = paraphraser(prompt, max_length=100, num_return_sequences=1)
    return out[0]['generated_text']

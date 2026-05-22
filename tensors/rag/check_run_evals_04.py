import openai
import os
from dotenv import load_dotenv
load_dotenv()
 
 
evals = [
  "What is Underwhelming Spatula?",
  "Who wrote 'Dubious Parenting Tips'?",
  "How long is Almost-Perfect Investment Guide?",
]
 
eval_answers = [
  "Underwhelming Spatula is a kitchen tool that redefines expectations by fusing whimsy with functionality.",
  "Lisa Melton wrote Dubious Parenting Tips.",
  "The Almost-Perfect Investment Guide is 210 pages long.",
]

def send_to_openai(message):
  openai.api_key = os.environ.get("OPENAI_API_KEY")
  completion = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": message}]
    )
  return completion.choices[0].message.content.strip()

def evaluate_generated_answer(
  expected_answer, 
  generated_answer):
  prompt = f"""Please evaluate the generated answer. If the generated answer provides the same 
  information as the expected answer, then return PASS. Otherwise, return FAIL and your reason for failing the answer.
  It is ok if the generated answer contains additional information that is not in the expected answer, 
  as long as the generated answer contains all the information included in the expected answer.
  Expected answer: {expected_answer} Generated answer: {generated_answer}"""
  response = send_to_openai(prompt)
  return response

from run_evals import evaluate_generated_answer


# def run_RAG(user_question):
#   return "IDKLOL"

import asyncio
from app import RAGChat

class fake_websocket:
  def print(self, input):
    pass

def run_RAG(user_question):
    answer = asyncio.run(RAGChat("", user_question, "hi@hi.com", fake_websocket()))
    return answer

def test_run_RAG():
 
  eval_questions = [
    "What is Underwhelming Spatula?",
    "Who wrote 'Dubious Parenting Tips'?",
    "How long is Almost-Perfect Investment Guide?",
    ]
 
  eval_answers = [
    "Underwhelming Spatula is a kitchen tool that redefines expectations by fusing whimsy with functionality.",
    "Lisa Melton wrote Dubious Parenting Tips.",
    "The Almost-Perfect Investment guide is 210 pages long.",
    ]
 
  generated_answers = []
  for question in eval_questions:
    answer = run_RAG(question)
    generated_answers.append(answer)
 
  for i in range(len(eval_questions)):
    result = evaluate_generated_answer(eval_answers[i],
      generated_answers[i])
    print(f"Eval answer: {eval_answers[i]}")
    print(f"Generated answer: {generated_answers[i]}")
    print(f"Result: {result}")
    assert "PASS" in result

#!/usr/bin/env python
# coding: utf-8

# # Lesson 2: RAG Triad of metrics

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()


# In[ ]:


from trulens_eval import Tru

tru = Tru()
tru.reset_database()


# In[ ]:


from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()


# In[ ]:


from llama_index import Document

document = Document(text="\n\n".\
                    join([doc.text for doc in documents]))


# In[ ]:


from utils import build_sentence_window_index

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)


# In[ ]:


from utils import get_sentence_window_query_engine

sentence_window_engine = \
get_sentence_window_query_engine(sentence_index)


# In[ ]:


output = sentence_window_engine.query(
    "How do you create your AI portfolio?")
output.response


# ## Feedback functions

# In[ ]:


import nest_asyncio

nest_asyncio.apply()


# In[ ]:


from trulens_eval import OpenAI as fOpenAI

provider = fOpenAI()


# ### 1. Answer Relevance

# In[ ]:


from trulens_eval import Feedback

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()


# ### 2. Context Relevance

# In[ ]:


from trulens_eval import TruLlama

context_selection = TruLlama.select_source_nodes().node.text


# In[ ]:


import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)


# In[ ]:


import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)


# ### 3. Groundedness

# In[ ]:


from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=provider)


# In[ ]:


f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)


# ## Evaluation of the RAG application

# In[ ]:


from trulens_eval import TruLlama
from trulens_eval import FeedbackMode

tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)


# In[ ]:


eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)


# In[ ]:


eval_questions


# In[ ]:


eval_questions.append("How can I be successful in AI?")


# In[ ]:


eval_questions


# In[ ]:


for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)


# In[ ]:


records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()


# In[ ]:


import pandas as pd

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]


# In[ ]:


tru.get_leaderboard(app_ids=[])


# In[ ]:


tru.run_dashboard()


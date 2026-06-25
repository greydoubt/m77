#!/usr/bin/env python
# coding: utf-8

# # Lesson 3: Sentence Window Retrieval

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()


# In[ ]:


from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()


# In[ ]:


print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])


# In[ ]:


from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))


# ## Window-sentence retrieval setup

# In[ ]:


from llama_index.node_parser import SentenceWindowNodeParser

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)


# In[ ]:


text = "hello. how are you? I am fine!  "

nodes = node_parser.get_nodes_from_documents([Document(text=text)])


# In[ ]:


print([x.text for x in nodes])


# In[ ]:


print(nodes[1].metadata["window"])


# In[ ]:


text = "hello. foo bar. cat dog. mouse"

nodes = node_parser.get_nodes_from_documents([Document(text=text)])


# In[ ]:


print([x.text for x in nodes])


# In[ ]:


print(nodes[0].metadata["window"])


# ### Building the index

# In[ ]:


from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)


# In[ ]:


from llama_index import ServiceContext

sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    # embed_model="local:BAAI/bge-large-en-v1.5"
    node_parser=node_parser,
)


# In[ ]:


from llama_index import VectorStoreIndex

sentence_index = VectorStoreIndex.from_documents(
    [document], service_context=sentence_context
)


# In[ ]:


sentence_index.storage_context.persist(persist_dir="./sentence_index")


# In[ ]:


# This block of code is optional to check
# if an index file exist, then it will load it
# if not, it will rebuild it

import os
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import load_index_from_storage

if not os.path.exists("./sentence_index"):
    sentence_index = VectorStoreIndex.from_documents(
        [document], service_context=sentence_context
    )

    sentence_index.storage_context.persist(persist_dir="./sentence_index")
else:
    sentence_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./sentence_index"),
        service_context=sentence_context
    )


# ### Building the postprocessor

# In[ ]:


from llama_index.indices.postprocessor import MetadataReplacementPostProcessor

postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)


# In[ ]:


from llama_index.schema import NodeWithScore
from copy import deepcopy

scored_nodes = [NodeWithScore(node=x, score=1.0) for x in nodes]
nodes_old = [deepcopy(n) for n in nodes]


# In[ ]:


nodes_old[1].text


# In[ ]:


replaced_nodes = postproc.postprocess_nodes(scored_nodes)


# In[ ]:


print(replaced_nodes[1].text)


# ### Adding a reranker

# In[ ]:


from llama_index.indices.postprocessor import SentenceTransformerRerank

# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)


# In[ ]:


from llama_index import QueryBundle
from llama_index.schema import TextNode, NodeWithScore

query = QueryBundle("I want a dog.")

scored_nodes = [
    NodeWithScore(node=TextNode(text="This is a cat"), score=0.6),
    NodeWithScore(node=TextNode(text="This is a dog"), score=0.4),
]


# In[ ]:


reranked_nodes = rerank.postprocess_nodes(
    scored_nodes, query_bundle=query
)


# In[ ]:


print([(x.text, x.score) for x in reranked_nodes])


# ### Runing the query engine

# In[ ]:


sentence_window_engine = sentence_index.as_query_engine(
    similarity_top_k=6, node_postprocessors=[postproc, rerank]
)


# In[ ]:


window_response = sentence_window_engine.query(
    "What are the keys to building a career in AI?"
)


# In[ ]:


from llama_index.response.notebook_utils import display_response

display_response(window_response)


# ## Putting it all Together

# In[ ]:


import os
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage


def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


# In[ ]:


from llama_index.llms import OpenAI

index = build_sentence_window_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    save_dir="./sentence_index",
)


# In[ ]:


query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)


# ## TruLens Evaluation

# In[ ]:


eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)


# In[ ]:


from trulens_eval import Tru

def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)


# In[ ]:


from utils import get_prebuilt_trulens_recorder

from trulens_eval import Tru

Tru().reset_database()


# ### Sentence window size = 1

# In[ ]:


sentence_index_1 = build_sentence_window_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=1,
    save_dir="sentence_index_1",
)


# In[ ]:


sentence_window_engine_1 = get_sentence_window_query_engine(
    sentence_index_1
)


# In[ ]:


tru_recorder_1 = get_prebuilt_trulens_recorder(
    sentence_window_engine_1,
    app_id='sentence window engine 1'
)


# In[ ]:


run_evals(eval_questions, tru_recorder_1, sentence_window_engine_1)


# In[ ]:


Tru().run_dashboard()


# ### Note about the dataset of questions
# - Since this evaluation process takes a long time to run, the following file `generated_questions.text` contains one question (the one mentioned in the lecture video).
# - If you would like to explore other possible questions, feel free to explore the file directory by clicking on the "Jupyter" logo at the top right of this notebook. You'll see the following `.text` files:
# 
# > - `generated_questions_01_05.text`
# > - `generated_questions_06_10.text`
# > - `generated_questions_11_15.text`
# > - `generated_questions_16_20.text`
# > - `generated_questions_21_24.text`
# 
# Note that running an evaluation on more than one question can take some time, so we recommend choosing one of these files (with 5 questions each) to run and explore the results.
# 
# - For evaluating a personal project, an eval set of 20 is reasonable.
# - For evaluating business applications, you may need a set of 100+ in order to cover all the use cases thoroughly.
# - Note that since API calls can sometimes fail, you may occasionally see null responses, and would want to re-run your evaluations.  So running your evaluations in smaller batches can also help you save time and cost by only re-running the evaluation on the batches with issues.

# In[ ]:


eval_questions = []
with open('generated_questions.text', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)


# ### Sentence window size = 3

# In[ ]:


sentence_index_3 = build_sentence_window_index(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index_3",
)
sentence_window_engine_3 = get_sentence_window_query_engine(
    sentence_index_3
)

tru_recorder_3 = get_prebuilt_trulens_recorder(
    sentence_window_engine_3,
    app_id='sentence window engine 3'
)


# In[ ]:


run_evals(eval_questions, tru_recorder_3, sentence_window_engine_3)


# In[ ]:


Tru().run_dashboard()


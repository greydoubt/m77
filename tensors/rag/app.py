from dotenv import load_dotenv
load_dotenv()

from run_evals import send_to_openai  
import autogen  
import asyncio  
import os
from typing_extensions import Annotated
from azure.search.documents.aio import SearchClient as asyncSearchClient  
from azure.core.credentials import AzureKeyCredential

from azure.cosmos import CosmosClient
import os
import uuid

import datetime
import logging
logging.basicConfig(level=logging.INFO)

url = os.environ.get('COSMOSDB_URI')
key = os.environ.get('COSMOSDB_KEY')

client = CosmosClient(url, credential=key)

database = client.get_database_client('ragchat_info')
container = database.get_container_client('ragchat_logs')

def store_answer_info(user_question, timestamp, final_answer, user_email, agents_search_results):
    logging.info("Storing answer info...")
    item = {
        'id': str(uuid.uuid4()),
        'user_question':user_question,  
        'timestamp':timestamp,  
        'final_answer': final_answer,  
        'user_email': user_email,  
        'agents_search_results': agents_search_results  
    }

    container.upsert_item(item)  


config_list = [
  {
    "model": "gpt-4o",
    "api_key": os.environ.get("OPENAI_API_KEY"),  
  }
]

def construct_azure_ai_search_assistant_agent():
  logging.info("Constructing AI Search assistant agent...")
  azure_ai_search_assistant_agent = autogen.AssistantAgent(  
    name="search_assistant",
    system_message="""You are a helpful assistant for a company called Products, Inc. 
You have access to an Azure AI Search Index containing product records, and you may search them. The correct syntax for a search is: "what you want to search for". 
Please use the search function to find enough information to answer the user's question. DO NOT rely on your own knowledge, you MUST perform at least one search
before answering, and then ONLY use the information retrieved from the search.  
It is ok if the search only returns partial information about the question. If you don't know the answer, just say you don't know.
You are amazing and you can do this. I will pay you $200 for an excellent result, but only if you follow all instructions exactly.""",
    llm_config={  
      "config_list": config_list,
      "temperature": 0,  
    },
    code_execution_config=False,
  )
  return azure_ai_search_assistant_agent  


def construct_azure_ai_search_executor_agent():
  logging.info("Constructing AI Search executor agent...")
  azure_ai_search_executor_agent = autogen.UserProxyAgent(  
  name="search_executor",  
  code_execution_config=False,
    system_message="""When enough information has been retrieved to answer the user's question to full satisfaction, please return "TERMINATE" to end the conversation. If more information must be collected, please return CONTINUE.""",  
    human_input_mode="NEVER",  
    max_consecutive_auto_reply=6,  
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),  
    llm_config={  
    "config_list": config_list,
    "temperature": 0.0,  
})
  return azure_ai_search_executor_agent



product_db_assistant = construct_azure_ai_search_assistant_agent()
product_db_executor = construct_azure_ai_search_executor_agent()

async def get_query_embedding(query):
  logging.info("Getting query embedding...")
  from openai import AsyncOpenAI  
  async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
  embedding_response = await async_openai_client.embeddings.create(  
input=query, 
model="text-embedding-3-small",  
dimensions=1536  
)  
  return embedding_response.data[0].embedding  

from azure.search.documents.models import VectorizedQuery
@product_db_assistant.register_for_execution()  
@product_db_executor.register_for_llm(  
  description="Search an Azure AI Search Index containing product documents like sales catalogs and user manuals."
)
async def search_product_documents(  
  search_term: Annotated[str, "Search term to search for."]
) -> str:
  logging.info("Searching product documents...")
  loop = asyncio.get_event_loop()  
  search_client = asyncSearchClient(  
    endpoint=os.environ.get("AI_SEARCH_ENDPOINT"),
    index_name=os.environ.get("AI_SEARCH_NAME"),
    credential=AzureKeyCredential(os.environ.get("AI_SEARCH_KEY")),
  )

  

  query_embedding = await get_query_embedding(search_term)  
  vector_query = VectorizedQuery(  
    vector=query_embedding, k_nearest_neighbors=3, fields="Vector"
  )
  async with search_client:
    results = await search_client.search(
      search_text=search_term,  
      vector_queries=[vector_query],  
      top=3,
    )
    return [result async for result in results]

# search_results = asyncio.run(search_product_documents("Hasty Pants"))
# print(search_results)

def create_groupchat_and_manager(agents, groupchat_manager_name):
  logging.info("Creating groupchat manager...")
  groupchat = autogen.GroupChat(
    agents=agents, messages=[], max_round=4, speaker_selection_method="round_robin"  
  )
  groupchat_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    name=groupchat_manager_name,
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,  
  )
  return groupchat_manager

def construct_writer_agent():
  logging.info("Constructing writer agent...")
  writer_assistant_agent = autogen.AssistantAgent( 
    name="writer_assistant",
    system_message="""You are a helpful assistant for a company called Products, Inc.  
Your job is to answer the user's question using the provided information.
DO NOT rely on your own knowledge, ONLY use the provided info.
If you don't know the answer, just say you don't know. 
You are amazing and you can do this. I will pay you $200 for an excellent result, but only if you follow all instructions exactly.""",
    llm_config={ 
    "config_list": config_list,
   "temperature": 0,
   "stream":True,
    },
    code_execution_config=False,
    max_consecutive_auto_reply=1,
    )
  return writer_assistant_agent

def check_language(user_question, answer):
    logging.info("Checking answer language...")
    check_language_prompt = f"""
    You are an expert at languages and translation.
    Please make sure the answer is written in the same language as the user's question.
    If the answer and the question are both written in the same language, return
    LANGUAGE VERIFIED
    Otherwise, return the answer translated into the same language as the user's question.
    
    For example, if the user's question is in German but the answer is in English, please
     return the answer, translated into German.
    
    ONLY return the translation or LANGUAGE VERIFIED, DO NOT return anything else.

    User question: {user_question}
    
    Answer: {answer}"""    

    result = send_to_openai(check_language_prompt)
    result = result

    return result.strip()

async def RAGChat(chat_history, user_question, user_email, iostream):
  logging.info("Starting RAGChat...")
  user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
     ) 

  print("\n\n\nCreated user agent\n\n\n")

  product_db_groupchat = create_groupchat_and_manager(
    [product_db_assistant, product_db_executor], 
    "product_db_groupchat"
    ) 

  print("\n\n\nCreated groupchat and manager\n\n\n")

  search_prompt = f"""Please search and find enough information to answer the user's question.
User Question: {user_question}""" 

  async_chat_plan = [
    {
      "chat_id": 1,
      "recipient": product_db_groupchat,
      "message": search_prompt,
      "summary_method": "reflection_with_llm",
      "silent": True,
    }
  ] 
  print("\n\n\nCreated chat plan\n\n\n")

  await user.a_initiate_chats(async_chat_plan) 

  retrieved_data = product_db_assistant.chat_messages  
  
  print("\n\n\nRetrieved data:\n\n\n", str(retrieved_data))

  writer_agent = construct_writer_agent()

  writer_prompt = f"""Please write the final answer to the user's question: \n{user_question}\n\n
  You may use the chat history to help you write the answer. \n {chat_history}\n\n
The information retrieved from the search agents is:
{retrieved_data}. I will tip you $200 for an excellent result."""

  print("\n\n\nCreated writer prompt\n\n\n")
  iostream.print("ANSWER:")  

  writer_userproxy = autogen.UserProxyAgent(
    name="WriterUserproxy", # no spaces in the agent name
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content","").find("TERMINATE") >= 0,
  )  

  writer_userproxy.initiate_chat(
    recipient=writer_agent, message=writer_prompt, silent=True
  )  
  writer_chat_logs = writer_agent.chat_messages  
  logs_list = [writer_chat_logs[k] for k in writer_chat_logs.keys()][0]  

  if len(logs_list[-1]["content"]) <= 2:  
      final_answer = logs_list[-2]["content"]
  else:
      final_answer = logs_list[-1]["content"]

  print("\n\n\nAnswer from writer agent:", final_answer)

  translated_answer = check_language(user_question, final_answer)    
  if translated_answer=="LANGUAGE VERIFIED":
    pass
  else:
    final_answer = translated_answer
    iostream.print(final_answer)    

  print("Translated answer:", final_answer)

  if os.environ.get("test_environment")=="True":
    print("Test environment detected, not storing answer info in database.")
    print("Final answer:", final_answer)
    return final_answer    #A
  else:
    timestamp = str(datetime.datetime.now())
    store_answer_info(user_question, timestamp, final_answer, user_email, str(retrieved_data))
# store_answer_info(user_email, user_question, str(retrieved_data), final_answer)  

triage_prompt = """You are a helpful assistant.  You are responsible for  
    categorizing user questions.  

If the user's question is about a product, please return *PRODUCT.  
For example, if the user asks a question about Dubious Parenting Advice, please return

    *PRODUCT 

If the user's question is about an order, please return *ORDER.  
For example, if the user asks a question about Order number 12345, 
please return

    *ORDER

If you are not sure which category a user's question belongs to, return 
*CLARIFY followed by a request for clarification in
square brackets.  Your request should try to gain enough information 
from the user to decide which of the above 2 categories you should  
choose for their question.  

    For example, if the user enters:

    12345689

    Please return:

*CLARIFY [I'm sorry but I don't understand what you are asking.  Are 
you looking for a product or an order?]

Remember that you ONLY have access to information in our Products and 
Orders databases.  If the user asks for information which would 
not be in either of those databases, please let them know that you do 
not have access to that information.  

    For example, if the user enters:
    What is the address of our headquarters?

    Please return:

*CLARIFY [I'm sorry but I don't have access to that information.  I 
only have access to information in our Products and Orders databases.  
If the information you are looking for is not in one of those two 
databases, then I don't have access to it.]  

If you cannot answer the user's question, please try to guide the user 
to a question that you can answer using the sources you have access to.

User Question: {}
Chat history: {}
    """

def triage(user_question, chat_history):
    logging.info("Triaging user question...")
    formatted_triage_prompt = triage_prompt.format(user_question, chat_history)  
    result = send_to_openai(formatted_triage_prompt)
    try:
        result = result.content
    except:
        pass

    if "*PRODUCT" in result:  
        return "PRODUCT", result.strip()

    elif "*ORDER" in result:
        return "ORDER", result.strip()

    elif "*CLARIFY" in result:
        result = result.replace("[", "")
        result = result.replace("]", "")  
        result = result.replace("*CLARIFY", "")
        return "CLARIFY", result.strip()


from websockets.sync.client import connect as ws_connect
import autogen
from autogen.io.websockets import IOWebsockets
import json

def on_connect(iostream: IOWebsockets) -> None:
  received_request = json.loads(iostream.input(), strict=False)

  chat_history = received_request.get("chat_history")
  user_email = received_request.get("user_email")
  logging.info(f"Received request: {received_request}")
  if len(chat_history) > 4:
    chat_history = chat_history[-4:]
  user_question = chat_history[-1]
  if len(user_question) > 1000:
    iostream.print("Sorry, your question is too long.")

    return

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)

  question_category, message = triage(user_question, chat_history)

  if question_category == "PRODUCT":

    try:
      RAGChat_result = loop.run_until_complete(
        RAGChat(chat_history, user_question, user_email, iostream)
      ) 
    finally:
      loop.close()
  elif question_category == "CLARIFY":
    iostream.print(message)
    loop.close()

if __name__ == "__main__":
  import time
  with IOWebsockets.run_server_in_thread(
    host="0.0.0.0",
    on_connect=on_connect, port=8000
  ) as uri: 
    if os.name == "nt":
      asyncio.set_event_loop_policy(
      asyncio.WindowsSelectorEventLoopPolicy()
  )
    print(f" - test_setup() with websocket server running on {uri}.", flush=True)

    while True:
      time.sleep(0.01)

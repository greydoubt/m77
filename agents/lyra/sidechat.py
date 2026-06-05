# 5 dec key manager integration
# 24 oct dynamodb call
# 5 nov azure switch
# 5 nov+ base64 switch


### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###

# for message handling:
import json
import base64

# for python stunts:
from typing import List

# for dynamodb functionality:
import boto3
from botocore.exceptions import ClientError


from datetime import datetime # possibly depricated!!!!

import time

# for all the kids in africa:
from lyra_plugins import *
from lyra import *

plugin_system_messages = macro_instructions # compatibility wrapper
master_system_prompt = macro_instructions["<<disclaimer>>"]

### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###

AZURE = True

import openai



def get_secret(secret_name = "openaikey"):

    
    region_name = "us-east-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
            
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    return secret


if not AZURE:
    openai.api_key = get_secret()
else:
    openai.api_type = "azure"
    openai.api_base = "https://q-ai-east-2.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = get_secret("azureopenaikey2")




### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###


# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('test_user_visits') 


### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###

# FROM MAIN CHAT (12 dec 23 version):


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
#
class ChatBot:
    def __init__(self, system=""):# add in endpoint tag for conditional and model choice
        self.system = system
        self.messages = []

        # getter setters for catching the last X message(s)
        # stream handler code

        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        #print("calling chatbot object")
        #print(f"chat history: {str(self.messages)}")
        
        # alter function parameters and insert previous history here
        self.messages.append({"role": "user", "content": message}) 
        
        print("messages appended")
        
        try:
            result = self.execute()
        except:
            print("execution failed!!")
        
        
        print("execution results:")
        print(result)
        
        self.messages.append({"role": "assistant", "content": result})
        
        return result
    
    def execute(self):

        #print("inside execute -- attempting now")
        try:

            print(f"completion for: {self.messages}")
            completion = openai.ChatCompletion.create(
                model=global_model_choice, # this is changed if using azure endpoint
                messages=self.messages,
                #temperature=0.2,
                temperature=1,
                stream=False
                )
            return ("completion for: {self.messages}", completion)
            #print(completion)

        except openai.error.Timeout as e:
            #Handle timeout error, e.g. retry or log
            print(f"OpenAI API request timed out: {e}")
            pass
        except openai.error.APIError as e:
            #Handle API error, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error, e.g. check network or log
            print(f"OpenAI API request failed to connect: {e}")
            pass
        except openai.error.InvalidRequestError as e:
            #Handle invalid request error, e.g. validate parameters or log
            print(f"OpenAI API request was invalid: {e}")
            pass
        except openai.error.AuthenticationError as e:
            #Handle authentication error, e.g. check credentials or log
            print(f"OpenAI API request was not authorized: {e}")
            pass
        except openai.error.PermissionError as e:
            #Handle permission error, e.g. check scope or log
            print(f"OpenAI API request was not permitted: {e}")
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error, e.g. wait or log
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except:
            print("unknown error with OpenAI endpoint")



        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # info: https://platform.openai.com/docs/guides/gpt/completions-response-format


        print(completion.usage) # need to broadcast this or have it passed back as a part of the response below to track usage

        #time.sleep(1) #deactivate or lambda hangs

        return completion.choices[0].message.content

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
#
class Message:
    def __init__(self, id: int, version_id: int, text: str, timestamp: str, sender: str, 
                 conversationTopic: str, tool: List[str], reference: List[str], 
                 token_cost: int, rating: int) -> None:
        self._id = id
        self._version_id = version_id
        self._text = text
        self._timestamp = timestamp
        self._sender = sender
        self._conversationTopic = conversationTopic

        # metadata section
        self._tool = tool
        self._reference = reference
        self._token_cost = token_cost
        self._rating = rating

    @classmethod
    def from_json(cls, json_string: str) -> 'Message':
        data = json.loads(json_string)
        metadata = data['metadata']
        return cls(data['id'], data['version_id'], data['text'], data['timestamp'], data['sender'], 
                   metadata['conversationTopic'], metadata['tool'], metadata['reference'], 
                   metadata['token_cost'], metadata['rating'])

    def to_json(self) -> str:
        return json.dumps({
            'id': self._id,
            'version_id': self._version_id,
            'text': self._text,
            'timestamp': self._timestamp,
            'sender': self._sender,
            'metadata': {
                'conversationTopic': self._conversationTopic,
                'tool': self._tool,
                'reference': self._reference,
                'token_cost': self._token_cost,
                'rating': self._rating
            }
        })
    
    # Getter and Setter for tool with type annotations
    @property
    def tool(self) -> List[str]:
        return self._tool

    @tool.setter
    def tool(self, tool: List[str]) -> None:
        self._tool = tool

    # Getter and Setter for reference with type annotations
    @property
    def reference(self) -> List[str]:
        return self._reference

    @reference.setter
    def reference(self, reference: List[str]) -> None:
        self._reference = reference

    # Getter and Setter for id
    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        self._id = id

    # Getter and Setter for version_id
    @property
    def version_id(self) -> int:
        return self._version_id

    @version_id.setter
    def version_id(self, version_id: int) -> None:
        self._version_id = version_id

    # Getter and Setter for text
    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        self._text = text

    # Getter and Setter for timestamp
    @property
    def timestamp(self) -> str:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp: str) -> None:
        self._timestamp = timestamp

    # Getter and Setter for sender
    @property
    def sender(self) -> str:
        return self._sender

    @sender.setter
    def sender(self, sender: str) -> None:
        self._sender = sender

    # Getter and Setter for conversationTopic
    @property
    def conversationTopic(self) -> str:
        return self._conversationTopic

    @conversationTopic.setter
    def conversationTopic(self, conversationTopic: str) -> None:
        self._conversationTopic = conversationTopic

    # Getter and Setter for token_cost
    @property
    def token_cost(self) -> int:
        return self._token_cost

    @token_cost.setter
    def token_cost(self, token_cost: int) -> None:
        self._token_cost = token_cost

    # Getter and Setter for rating
    @property
    def rating(self) -> int:
        return self._rating

    @rating.setter
    def rating(self, rating: int) -> None:
        self._rating = rating
# end of Message class definition

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# this replaces the single function below
class UserVisitManager:
    def __init__(self, table_name='test_user_visits'):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def update_user_visit(self, user):
        """
        Update the last_visit timestamp and increment the visits count for a user.
        If the user doesn't exist, create a new entry with initial state.
        """


        # UPDATE: add token count and whatever other KPIs



        response = self.table.update_item(
            Key={'user_id': user},
            UpdateExpression="SET last_visit = :last_visit ADD plugin_calls :increment",
            #UpdateExpression="SET last_visit = if_not_exists(last_visit, :initial_date) ADD visits :increment",
            ExpressionAttributeValues={
                #':last_visit': datetime.utcnow().isoformat(),
                ':last_visit': int(time.time()),
                ':increment': 1
            },
            ReturnValues="ALL_NEW"
        )
        return response['Attributes']



##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 



### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###


# deprecate this asap:
# function: dynamodb integration 
def update_user_visit(user):
    """
    Update the last_visit timestamp and increment the visits count for a user.
    """
    response = table.update_item(
        Key={'user_id': user},
        UpdateExpression="SET last_visit = :last_visit ADD visits :increment",
        ExpressionAttributeValues={
            #':last_visit': datetime.utcnow().isoformat(),
            ':last_visit': int(time.time()),
            ':increment': 1
        },
        ReturnValues="ALL_NEW"
    )
    return response['Attributes']

### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###

def lambda_handler(event, context):


    ###############################################################


    # LLM specifics
    history_engram = ""
    llm_response = ""
    system_prompt = ""
    final_token_count = 0 

    flag_gen_subject = False
    flag_lockout_copywriter = False

    ADMIN_MODE = False
    VERBOSE_MODE = False

    history_engram = ""
    
    engine_choice = "gpt3516k"



    # Billing Specifics
    billable_action = True # changes if in admin
    allowable_action = False



    # DB Ops




    request_context = event.get('requestContext', {})

    authorizer = request_context.get('authorizer', {})
    claims = authorizer.get('claims', {})

    user_id = claims.get('principalId')  # This will be the user's unique identifier (sub)


    # Extract user name
    user_name = claims.get('cognito:username')

    manager = UserVisitManager() # this doesnt work unless the auth layer is applied to API gateway


    # CHECK DB HERE



    # Extract user group information
    user_groups = claims.get('cognito:groups', [])

    if user_groups == "ADMIN":
        print("Admin user mode [LOGGING ENABLED]")
        VERBOSE_MODE = True
        ADMIN_MODE = True
        billable_action = False # changes if in admin
        allowable_action = True

    # this is a temp fix:
    def s_print(message):
        # suppressive print / use to retrofit old stuff
        if VERBOSE_MODE:
            print(message)
        else:
            print("PROTECTED DATA -- OMIT FROM LOGS")



    ###############################################################


    # Extract the request body from the event
    body = event.get('body', '')
    headers = event.get('headers', '')
    request_context = event.get('requestContext', {})

    decoded_body = base64.b64decode(body).decode('utf-8')
    
    # Extract substring
    #payload = decoded_body[10:-2]

    user_message = Message.from_json(decoded_body[10:-2])

    s_print("------------------------")
    s_print(f"EVENT: {event}")
    s_print(type(event))
    s_print(f"EVENT: {event['body']}")
    s_print(type(event['body']))
    s_print(f"EVENT: {event['body']}")
    s_print(type(event['body']))
    s_print(f"PATH: {event['path']}")
    s_print("------------------------")

    s_print("------- DATA OPS -------")

    #user = "U12347" #update to use dynamodb for userbase

    # Update the DynamoDB item
    #responseupdated_attributes = update_user_visit(user)

    s_print("------- END DATA -------")
    
    #tool = event.get('tool', '')
    #body = event.get('body', '')
    #print(f"BODY: {body}")
    #print(type(body))
    
    #tools = user_message.tool
    #references = user_message.reference

    tools = [base64.b64decode(t).decode('utf-8') for t in user_message.tool]
    references = [base64.b64decode(r).decode('utf-8') for r in user_message.reference]

    try:
        new_conversation_topic = base64.b64decode(user_message.conversationTopic).decode('utf-8')
    except:
        new_conversation_topic = user_message.conversationTopic

    tool_reference_pairs = [(t, r) for t, r in zip(tools, references)]

    s_print(f"Tools and variables: {tool_reference_pairs}")

    
    #tool = d_body["tool"]
    #inputs = d_body['inputs']
    #print(f"tool: {tool}, inputs: {inputs}") # inputs is a dict
    
    #d_inputs = json.loads(inputs)
        # Tool messages: verified / updated 15 Sep 2023
    # deck_flow 👺
    # deck_inconsistencies 👺
    # action_title 👺
    # formatting_inconsistencies👺
    # ideate_layouts👺
    # academic_citation_formatting👺
    # executive_summary👺
    # grammar_check👺
    # feedback_sentiment👺
    # feedback_clarity👺
    # feedback_suggestions👺
    # questions_to_ask👺
    # academic_abstract👺
    # academic_conclusion👺
    # academic_ideate_discussion👺
    # academic_insert_toc👺
    # condense_text👺
    # enrich_text👺
    # transform_to_bullets👺
    # transform_to_paragraphs👺
    # rewrite_content👺
    # fix_grammar👺
    # text_transformation👺*

    decoded_user_content = base64.b64decode(user_message.text).decode('utf-8')
    user_content = f'Text input: """{decoded_user_content}"""'
    
    
    s_print(f"user_content: {user_content}")

    system_content = ""



    for tool, reference in tool_reference_pairs:


        s_print(f"attempting tool: {tool}" )
        s_print(f"w/ variable: {reference}" )


        if tool == "chatbot":
            s_print(f"TOOL OVERRIDE: MODEL CHOICE: {reference}")
            engine_choice = reference # needs a flag from the DB check

        elif tool == "<<quantify_metric>>" or tool == "<<excel_quantify_metric>>":
            print("TOOL OVERRIDE: switching to gpt4")
            engine_choice = "gpt4"


        elif tool == "chat_history":
            s_print("Handling chat_history..." + reference)
            history_engram = reference

            s_print("<HISTORY NOT IMPLEMENTED YET>")

            # this is inactive

        elif tool in plugin_system_messages:
            system_content += plugin_system_messages.get(tool)

            try: 
                system_content += plugin_system_messages[tool].format(variable=reference)
            except:
                system_content += system_content + " " + reference


        else:
            system_content += tool + " " + reference



            """try:
                                    
                                                    system_content += plugin_system_messages.get(tool)
                                            
                                                    s_print(f"getting tool: {system_content}" )
                                                    try:
                                                         
                                                        # all of this code is a huge work-around when the variable could just be    AFTER  the string / prompt. leaving here for now until the standard is formalised more
                                            
                                                    except Exception as e:
                                                        
                                        
                                                except Exception as e:
                                            
                                                    print("error getting tool")
                                            
                                                    return {
                                                        'statusCode': 501,
                                                        'message': "Tool value mangled or does not exist."
                                                    }"""


    

    # this adds the disclaimer to the system prompt
    system_content = system_content + master_system_prompt

    s_print(f"Sending to LLM [SYSTEM]: {system_content}")
    s_print(f"Sending to LLM [USER]: {user_content}")

    


    

        
    # if keep    manager.update_user_visit(user_name) BEFORE the LLM call so we charge even if the call fails -- but then we wont know the token count in advance!!!
    

    # ACTIVE CHANGES TO MAKE:
    
    # disable and replace with OOP version in order to use chat history:
    completion = openai.ChatCompletion.create(
        engine=engine_choice,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2, # note this is not the same as chat or OOP
                    #max_tokens=350,
                    #top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None)
            

    


    
    s_print(f"Completion from LLM: {completion}")

    llm_full_response = {
        "id": user_message.id,
        "version_id": user_message.version_id,
        "text": completion.choices[0].message.content,
        "timestamp": time.time(),#str(datetime.now()),#update to unix time stamp
        "sender": "q",
        #"conversationTopic": user_message.conversationTopic,#
        "conversationTopic": new_conversation_topic,
        "tool": user_message.tool, 
        "reference": user_message.reference,
        "token_cost": completion.usage.total_tokens, #update this later
        "rating": 0
    }

    manager.update_user_visit(user_name) # add model cjo



    api_response = Message(**llm_full_response).to_json()




    response_body = {
        'message': completion["choices"][0]["message"]["content"]
    }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(response_body)
        #'response': completion["choices"][0]["message"]["content"]
    }











### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ###
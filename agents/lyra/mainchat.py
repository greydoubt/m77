##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

import json
import time
from datetime import datetime

import base64
import random
import string

import logging

import openai 

import boto3 # need to reactivate this for DB integration
from botocore.exceptions import ClientError
from typing import List
from lyra import * # tenet internal tools and system prompts

master_system_prompt = macro_instructions["<<disclaimer>>"]

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

#openai.api_key = get_secret()

openai.api_type = "azure"

openai.api_base = "https://q-ai-east-2.openai.azure.com/" # engine="test_deployment" etc

openai.api_version = "2023-07-01-preview"

openai.api_key = get_secret("azureopenaikey2")



##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

#global_model_choice = "gpt-4"
#global_model_choice = "gpt-3.5-turbo-16k"
#global_model_choice = "gpt-3.5-turbo"
global_model_choice = "gpt3516k"

# with azure they are called engines, not models:


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 



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
        print("calling chatbot object")
        #print(f"chat history: {str(self.messages)}")
        
        # alter function parameters and insert previous history here
        self.messages.append({"role": "user", "content": message}) 
        
        print("messages appended")
        
        try:
            result = self.execute()
        except:
            print("execution failed!!")
        
        
        #print("execution results:")
        #print(result)
        
        self.messages.append({"role": "assistant", "content": result})
        
        return result
    
    def execute(self):
        print("inside execute -- attempting now")
        try:
            #print(f"completion for: {self.messages}")
            completion = openai.ChatCompletion.create(
                #model=global_model_choice, # this is changed if using azure endpoint
                engine=global_model_choice,
                messages=self.messages,
                #temperature=0.2,
                temperature=1,
                stream=False
                )
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
#
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
            UpdateExpression="SET last_visit = :last_visit ADD visits :increment",
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
#



##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

def lambda_handler(event, context):

    # not needed yet:
    #logging.basicConfig(level=logging.INFO)

    # LLM specifics
    history_engram = ""
    llm_response = ""
    system_prompt = ""
    final_token_count = 0 

    flag_gen_subject = False
    flag_lockout_copywriter = False


    # Extract the request body from the event
    body = event.get('body', '')
    headers = event.get('headers', '')
    request_context = event.get('requestContext', {})

    decoded_body = base64.b64decode(body).decode('utf-8')
    
    
    
    
    
    
    #d_body = json.loads(decoded_body)


    # for plugins compatiblity:
    # 
    

    # Extract substring
    payload = decoded_body[10:-2] # this can be safely removed but the json is one step lower


    ####### ##### 
    



    # USER AUTH SECTION - 3.0 move to outside helper function
    # 
    # Get the authorizer information

    authorizer = request_context.get('authorizer', {})
    claims = authorizer.get('claims', {})

    user_id = claims.get('principalId')  # This will be the user's unique identifier (sub)


    # Extract user name
    user_name = claims.get('cognito:username')

    manager = UserVisitManager()

    manager.update_user_visit(user_name) # CRUD OPS


    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 


    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 



    
    # Extract user group information
    user_groups = claims.get('cognito:groups', [])

    # LiveGroup, TestGroup
    # Route TGroup to the inactive pipeline
    # Route LiveGroup to the active

    # if-else check-point
    print(f"USER GROUPS: {user_groups}")
    #print(type(user_groups))

    VERBOSE_MODE = False 


    if user_groups == "ADMIN":
        print("Admin user {}")
        VERBOSE_MODE = True

    # this is a temp fix:
    def s_print(message, mode = VERBOSE_MODE):
        # suppressive print / use to retrofit old stuff
        if mode:
            print(message)
        else:
            print("PROTECTED DATA -- OMIT FROM LOGS")






    if VERBOSE_MODE:
        
        print("Decoded: " + decoded_body)
        
        print("raw text:")
        
        print(payload)
        
        print(type(payload))
    
        # Log the full body to CloudWatch logs
        print(f"Full Request Headers {datetime.now()}: {headers}")
    
        # Log the full body to CloudWatch logs
        print(f"Full Request Body {datetime.now()}: {body}")
        
        authorization_token = headers.get('Authorization', '')
        
        print(f"Auth field: {authorization_token}")

    else:
        pass
    
    access_tier = ""
    post_message = ""

    if user_groups == "BLOCKED":#"":

        #response = f"Access Tier: {access_tier} UNKNOWN.  Request body has been logged to CloudWatch logs {datetime.now()}. USER: {user_name} USERGROUP: {user_groups}."
        return {
            'statusCode': 401,
            'body': f'User {user_name} not associated with access group policy.'
            }

        
    elif user_groups == "TestGroup":
        access_tier = "INACTIVE"
        # route to basic access / give info on how to sign up / show an ad

        #response = f"Access Tier: {access_tier}.  Request body has been logged to CloudWatch logs {datetime.now()}. USER: {user_name} USERGROUP: {user_groups}."

        return {
                    'statusCode': 403,
                    'body': f'User: {user_name}; Group {user_groups} does not have access to endpoint.'
                    }

        # end of free user access code
        
    elif user_groups == "ADMIN" or user_groups == []:#"LiveGroup" or "DemoGroupNov1" or "DemoGroupNov2" or "Staff" or "Development":
        access_tier = "ACTIVE"



        #service_flag = True
        response = "valid user" # place-holder

        final_payload = filter_unicode(payload) # move this / this was required for SQS but might not be important now
            
        s_print(f"preparing to send to LLM: {final_payload}") # move this?
        
        #try:
        user_message = Message.from_json(final_payload)
        s_print(f"User message: {user_message}")



        tools = user_message.tool

        references = user_message.reference


        tools = [base64.b64decode(t).decode('utf-8') for t in user_message.tool]
        references = [base64.b64decode(r).decode('utf-8') for r in user_message.reference]

        tool_reference_pairs = [(t, r) for t, r in zip(tools, references)]

        # immediately sort the list of tuples

        tool_reference_pairs = sort_by_hierarchy(tool_reference_pairs, hierarchy_list)


        # tooling code goes here
        #user_prompt = user_message.text
        user_prompt = base64.b64decode(user_message.text).decode('utf-8')
        try:
            new_conversation_topic = base64.b64decode(user_message.conversationTopic).decode('utf-8')
        except:
            new_conversation_topic = user_message.conversationTopic

        pre_prompt = " "
        post_prompt = " "

        s_print("check if tool is inside macro instructions:")
                        

        try:
            s_print(macro_instructions[user_message.tool[0]])
            
            system_prompt += macro_instructions.get(user_message.tool[0]) + user_message.reference[0]
        except:
            pass


        for tool, reference in tool_reference_pairs:

            match tool.lower():

                case "chatbot":
                    s_print("Handling chatbot..." + reference)
                    global global_model_choice
                    if reference == "gpt4":
                        global_model_choice = "gpt4"

        
                case "chat_history":
                    s_print("Handling chat_history..." + reference)
                    history_engram = reference

                case "autotool":
                    s_print("Handling autotool..." + reference)
                    pre_prompt += global_tools_desc

                
                case "generate_subject":
                    s_print("Handling generate_subject..." + reference)
                    flag_gen_subject = True
                    pre_prompt += f_gen_subject() 
                    
        
                case "aboutq":
                    s_print("Handling aboutq..." + reference)
                    pre_prompt += f_about_q()
                    
                case "aboutme":
                    pre_prompt += reference
                    
                    
                case "persona_generate":
                    # stub for new tool
                    pre_prompt += """Your task is to evaluate the persona and writing style of the author. Follow these steps:
Step 1: <Identify the author's MBTI>.
Step 2.1: <Identify the 3 most distinct descriptors of what makes this writing style unique.>
Step 2.2: <Identify the 3 most accurate and useful descriptors of this writing style.>
Step 2.3: <Identify the 4 most meaningful descriptors in order to replicate the writing style of this author.>
Step 3: <Redact and include 3 example sentences that are the best examples of this author's distinct writing style.>

Use the following format: \"\"\"MBTI: [4-char]. {word}, {word}, {word}, {word}, {word}, {word}, {word}, {word}, {word}, {word}. Example excerpts: '{sentence}'. '{sentence}'. '{sentence}'."""
                
                case "persona_update":
                    # stub for new tool
                    # for now just generate a new persona
                    pass

                case "persona_apply":
                    # stub for new tool
                    pre_prompt += reference
        

                case "teachq" | "instruction":
                    print("Handling teachq..." + reference)
                    
                    # this check if it is a premade (ie wrapped in << >>)
                    temp_ref_inst = replace_macros(reference, macro_instructions) 

                    pre_prompt += f"\nSpecial instructions: {temp_ref_inst}\n"#

                case "genai_improve" | "generate_improved_img_prompt":
                    s_print("Handling genai_improve..." + reference)

                    # changed this to a function call
                    pre_prompt += f_generate_improve_img_prompt(reference) # check this to make sure it pulls from the tag-macros list


                #### MISC
                case "app_integration":
                    s_print("Handling app_integration..." + reference)
                    # this is the junction for when calling the other EPs on base URL
                    # this is reserved / not used
                    pass
        



                ##### FORMAT LAYER
                case "format":
                    s_print("Handling Format..." + reference)
                    temp_ref_form = replace_macros(reference, macro_formats) 
                    post_prompt += f"Format: {temp_ref_form}\n"#

                case "tone":
                    s_print("Handling Tone..." + reference)
                    post_prompt += f"Add tone: {reference} \n"#
        
                case "length":
                    s_print("Handling Length..." + reference)
                    post_prompt += f"Length of response: {reference} \n"#
        
                case "ftl_other":
                    s_print("Handling ftl_other..." + reference)
                    post_prompt += f"Other style instructions: {reference} \n"#





                #### USER LEVEL
                case "chat_with_sources":
                    s_print("Handling chat_with_sources..." + reference)


                    #global_model_choice = "gpt-3.5-turbo-16k"
                    #s_print("switching LLM")

                    s_print("Source: ")
                    s_print(reference)


                    user_prompt += f_chat_with_sources(str(reference))

        


                case _:
                    s_print(f"User-defined tool: {tool}")
                    sub_tool = replace_macros(tool, macro_instructions) 
                    pre_prompt += f"User defined special instructions: {sub_tool} using {reference}"


        

        
        system_prompt += pre_prompt + post_prompt + master_system_prompt
        

        # tool ordering after processing tools
        #s_print(f"System prompt: {system_prompt}\n\n\n")


        q = ChatBot(system_prompt)
        
        cleaned_history = ""
        
        if history_engram:
            cleaned_history = history_engram.replace('`', '"')#.replace('\n', '\\n').replace('\r', '\\r')

        else:
            pass
            #cleaned_history = "[{`role`:`user`,`content`:`SGk=`},{`role`:`assistant`,`content`:`SGVsbG8hIEkgYW0gUSAtLSBob3cgbWF5IEkgYXNzaXN0IHlvdSB0b2RheT8=`},{`role`:`user`,`content`:`VGVsbCBtZSBhYm91dCB5b3VyIGNhcGFiaWxpdGllcw==`},{`role`:`assistant`,`content`:`TXkgYWJpbGl0aWVzIGNhbiBiZSBleHBsYWluZWQgdGhyb3VnaCB0aGUgYWJvdXRxIHRvb2wsIGFuZCBhdXRvdG9vbCBpcyBhbHNvIG9uIHRoZSB0YWJsZSE=`}]".replace('`', '"')
        
        s_print(f"History: {cleaned_history}")

        # Append messages to the self.messages list using the specified format
        #for message in json.loads(cleaned_history):
        try:
            for message in json.loads(str(cleaned_history)):
                role = message['role']
                # 7 nov chat history stability change
                content = base64.b64decode(message['content']).decode('utf-8')
                content = content.replace('\n', '\\n').replace('\r', '\\r')
    
                q.messages.append({"role": role, "content": content})
        except:
            s_print("no history sent!!!")


        # try / catch with non-200 response for LLM failures]
        #print("USER PROMPT:")
        #s_print(user_prompt)
        
        s_print(f"Sending to LLM [SYSTEM]: {system_prompt}")
        s_print(f"Sending to LLM [HISTORY]: {cleaned_history}")
        s_print(f"Sending to LLM [USER]: {user_prompt}")
       
        
        # activate for debugging:
        # print(f"TOKEN COST (EST FROM CLIENT: {user_message.token_cost}")


        try:
            #llm_api_response = q(user_prompt)
            # def beta_dither_text_v2(text, MAX=3000, max_percentage=0.50)
            # def dither_text(text, MAX=2000)
            
            #llm_api_response = q(beta_dither_text_v2(user_prompt,12000))
            
            
            # expect to wait 1-20 seconds for the LLM response
            
            #print(llm_response)
            response = q(user_prompt)
            s_print(response)
            #
            #

        except Exception as e:
            print(f'Error with LLM endpoint: {e}')
            #llm_response = f'Error with LLM endpoint: {e}'

            return {
                'statusCode': 502,
                'body': f'Error with endpoint: {e}. Message exceeds context window.'
            }
            #



    else:
        # change this to an error
        response = f"Access Tier: UNKNOWN.  Request body has been logged to CloudWatch logs {datetime.now()}. USER: {user_name} USERGROUP: UNKNOWN."
        return {
            'statusCode': 403,
            'body': f'User: {user_name} is not in access plan. Contact sales to access endpoint.'
                    }

    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    #### check if valid access then send to LLM after checking tools
    # tools check 
    # LLM call
    # error-handling

    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

    # post-process LLM response and add to message

    s_print(f"Completion from LLM: {response}")
    if flag_gen_subject:
        user_message.conversationTopic = response


    # Sample data in a dictionary
    llm_full_response = {
        "id": user_message.id,
        "version_id": user_message.version_id,
        "text": response,
        "timestamp": str(datetime.now()),#update to unix time stamp
        "sender": "q",
        #"conversationTopic": user_message.conversationTopic,#
        "conversationTopic": new_conversation_topic,
        "tool": user_message.tool, 
        "reference": user_message.reference,
        "token_cost": 0, #update this later
        "rating": 0
    }


    api_response = Message(**llm_full_response).to_json()


    
    # Example response:
    response_body = {
        'message': api_response
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(response_body)
    }

# update the non-200 codes to follow this format



##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
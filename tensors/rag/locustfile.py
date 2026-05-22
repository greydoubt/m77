import json, time, websocket    
from locust import User, task, between, events    

HOST = "wss://your-web-app-url"    
PAYLOAD = {                        
    "email": "tester@example.com",
    "chat_history": ["user_question: What is Dubious Parenting Advice?"]
}

def fire(name, start, length=0, exc=None):    
    events.request.fire(
        request_type="WS",
        name=name,
        response_time=int((time.time() - start) * 1000),
        response_length=length,
        exception=exc)

class ChatUser(User):          
    wait_time = between(1, 3) 

    def connect(self):
        t = time.time()
        try:
            self.ws = websocket.create_connection(HOST, timeout=15)    
            fire("connect", t)
            self.ws.send(json.dumps(PAYLOAD))    
        except Exception as e:
            fire("connect", t, exc=e); raise
    def on_start(self): self.connect()    

    @task    
    def chat(self):    
        try:
            t = time.time(); msg = self.ws.recv()        
            fire("recv", t, len(msg))                    
            self.ws.send(json.dumps(PAYLOAD))            
        except Exception:                                
            self.ws.close(); self.connect()              

    def on_stop(self): self.ws.close()    

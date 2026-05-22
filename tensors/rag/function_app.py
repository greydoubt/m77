import azure.functions as func
import logging

import os
from azure.cosmos import CosmosClient
from dotenv import load_dotenv
from uuid import uuid4
load_dotenv()

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
cosmos_client = CosmosClient(
url = os.environ.get("COSMOSDB_ENDPOINT"),
credential = os.environ.get("COSMOSDB_KEY")
)
print("Cosmos client created")
feedback_db = cosmos_client.get_database_client(os.environ.get("COSMOSDB_DATABASE_NAME"))
feedback_container = feedback_db.get_container_client(os.environ.get("COSMOSDB_CONTAINER_NAME"))
print("Cosmos container connected")
print("ran the app thing")
logging.info("Ran the app thing")

@app.function_name(name="cosmosdb-func")
@app.route(route="cosmosdb_func", methods=["POST", "GET"])
def store_user_feedback(req: func.HttpRequest) -> func.HttpResponse:
    question = req.params.get("question")
    answer = req.params.get("answer")
    email = req.params.get("email")
    rating = req.params.get("rating")
# (include a generated image of thumb direction)
    reason = req.params.get("reason")

    item = {"question": question,
        "answer": answer,
        "rating": rating,
        "reason": reason,
        "email": email,
        "id": str(uuid4())
    }
    try:
        cosmosdb_response = feedback_container.upsert_item(item)
    except Exception as e:
        logging.error(f"Error storing feedback: {e}")
        return func.HttpResponse(
            "An error occurred while storing feedback.",
            status_code=500
        )
    return func.HttpResponse(
        "Successfully stored feedback.",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        }
    )

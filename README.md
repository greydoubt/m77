# HTML2Vec
Converts list of URLs to salient features for ML tasks

# THIS LIBRARY WILL DOWNLOAD THE ENTIRE INTERNET. IT CAN GET YOU BANNED, ARRESTED, DEPORTED, ETC




<img width="1021" height="373" alt="Screenshot 2026-06-04 at 18 44 59" src="https://github.com/user-attachments/assets/cc8928f7-c0a2-494f-ba82-6aa2374a052f" />


Q=K=V: single projection for all three roles (PSPACE Memory Optimisation)

Q=K–V: shared query and key, separate value (symmetric attention, however this breaks directionality and thus NP-Hard)

Q–K=V: separate query, shared key and value (asymmetric, hence directional vector and scalar NP-Complete, NOT NP-Hard, thus P vs NP speedup gain)



## Data UI/UX: The Jupyter & Sekka Satvrnos: The Notebook Hack

================



# Jupyter Notebook Extension: User Interaction Logger

Jupyter Notebook Extension to track user interaXtion. 

## Set up

### Requirements

-   Ensure Jupyter Notebook is installed and can be launched from the terminal or command prompt. 
    To check this, execute ```jupyter --paths``` from either your terminal or command prompt. If the command is not found, add
    jupyter to your PATH. 

-   Ensure that *jupyter notebook extensions manager* is installed.
    If not, follow the [directions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) and install it. 

-   Ensure that there is a jupyter notebook configuration file in your file system. When you execute ```jupyter --paths``` in the 
    command line, you should see a list of possible directories which can contain the config file. Check to see if you can find
    ```jupyter_notebook_config.json``` AND/OR ```jupyter_notebook_config.py``` in those directories. If you cannot, then execute: ```jupyter notebook --generate-config```. Some versions of Jupyter Notebook will include the python file and not the JSON config file. The py file can be configured
    for our purposes in this project. 

### "Installing" extensions

-   Using the ```jupyter --paths``` command as a guide, identify the directory containing ```nbextensions```. If none of the paths listed contain nbextensions, 
    you might have to perform a machine level search for the directory. You might find multiple directories under the same name. The directory we want will have a
    list of subdirectories of Jupyter Notebook extensions installed through the nbextentions command we executed above. Add the CaptureLogs folder from this
    repo under ```nbextensions```. You will know that we have chosen the right directory when 'CaptureLogs' appears as one of the extensions to enable as part of
    the nbextensions tab in the Jupyter Notebook home page. 

-   Append content from config/jupyter_notebook_config.json to ```jupyter_notebook_config.json```. If this file is not present in your config directory,
    then, uncomment the commented lines in config/jupyter_notebook_config.py before appending those lines to ```jupyter_notebook_config.py```. If the json
    file does exist, only append the uncommented lines to ```jupyter_notebook_config.py```. You should find a py file in one of the config paths listed by jupyter
    --paths command. 

-   Using the ```jupyter --paths``` command as a guide, identify the directory containing jupyter libraries. Add the SaveLogs folder from this repo under 
    the site-packages directory with other jupyter libraries. If you have difficulty identifying this directory, try running jupyter notebook from 
    the command line. If you find a 'SaveLogs module not found' error, that stack trace will identify the default path for the jupyter notebook libraries. 
    Make sure to add SaveLogs directly under site-packages in that path. 
    
-   You will need to execute the follow: `jupyter contrib nbextension install --user` if you are install Jupyter Notebook extensions manager for the first time. 

-   Launch Jupyter Notebook from your command line. In the file explorer view, you should see an 'nbextensions' tab. Select the tab, and
    find 'Capture Logs' extension among the list of other available extensions. Select Capture Logs and enable the extension. Restart your
    Jupyter Notebook. 

### Checking the installation

-   Launch jupyter notebook from the command line. You shouldn't see any errors or exceptions regarding SaveLogs in the command line. 
-   Create or edit notebooks as you normally would. 
-   Open your browser's developer tools and view the console. You should see logs labeled with an '[evt]' tag. This is the data being logged
    by our extension.
-   You should also see a '[notebook name]'_log.json file in the same directory as your notebook. This file is the archive of all your user interaction. 

=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-


<img width="1359" height="763" alt="Screenshot 2026-06-22 at 09 02 34" src="https://github.com/user-attachments/assets/ff166e29-7b56-4aad-9588-d04b420eab42" />




## Deep Large Language Model Learning for AI/ML REhazxe Module 11


STEP -1: INSTALL SOME STUFF
```
import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

```



Obtain an example dataset

To demonstrate how to save and load weights, you'll use the MNIST dataset. To speed up these runs, use the first 1000 examples:

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


Define a simple sequential model circuit
```
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model
```



<img width="530" height="344" alt="Screenshot 2025-02-25 at 05 50 32" src="https://github.com/user-attachments/assets/ec0040b1-54bc-474f-8bd4-91e04e01dce4" />



Create a basic model instance
```
model = create_model()
```

Display the model's architecture
```
model.summary()
```


## Checkpoint callback usage to save weights [only] during training:
```
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
```

_Create a callback that saves the model's weights_
```
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
```

Train the model with the new callback
```
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training
```

## __$CAVEAT VEKTOR: ## This may generate warnings related to saving the state of the optimizer._
## __These warnings (and similar warnings throughout this notebook)_
## __are in place to discourage outdated usage, and can be ignored._


=-=-=--==-=-=-


<img width="489" height="277" alt="Screenshot 2025-02-25 at 05 16 31" src="https://github.com/user-attachments/assets/2c8aa32e-c87e-480e-bc91-fc28f63191d0" />

=-=-=--==-=-=-



_This creates a single collection of TensorFlow checkpoint files _ that are updated at the end of each epoch:_ 
```
os.listdir(checkpoint_dir)
```

As long as two models share the same architecture you can share weights between them. Thus when restoring a model just from weights, one creates a model with the same architecture as the original model and then set its weights as follows:


Rebuild a fresh, untrained model and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy or "House Odds")
```
model = create_model()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
```


# load the weights from the checkpoint and re-evaluate:
```
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
```


=-=-=--==-=-=-=-=-=--==-=-=-



<img width="487" height="282" alt="Screenshot 2025-02-25 at 05 16 26" src="https://github.com/user-attachments/assets/4e773a63-a9d3-4c1f-8292-b9e2e7666164" />




=-=-=--==-=-=-


# Model Checkpoint callback options (MOEC-OS, aka MOCCOS, aka Mach-BOOGERS)

_Include the epoch in the file name (uses `str.format`)_
```
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32
```

_Calculate the number of batches per epoch_
```
import math
n_batches = len(train_images) / batch_size
n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer
```

_Create a callback that saves the model's weights every 5 epochs_
```
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*n_batches)
```

_Create a new model instance_
model = create_model()


_Save the weights using the `checkpoint_path` format_
```
model.save_weights(checkpoint_path.format(epoch=0))
```

_Train the model with the new callback_
```
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)
```

To review the resultant checkpoints and choose the latest one from the batch of artifacts one simply lists the directory as such:
```
os.listdir(checkpoint_dir)

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
```
* Note: The default TensorFlow format only saves the 5 most recent checkpoints.



To test, reset the model, and load the latest checkpoint, just create a new model instance
```
model = create_model()
```


_Load the previously saved weights_
```
model.load_weights(latest)
```

_Re-evaluate the model_
```
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
```


The artefact or artifacts (artifices) that result will thus give you Artifice-ial Intelligence, or, the Intelligence an Artifice makes, can read, and can process

Following this metaphor in the most literal sense, the above code accordingly stores the weights to a collection of checkpoint-formatted files that contain only the trained weights in a binary format. Checkpoints contain:

_One or more shards that contain your model's weights._

_An index file that indicates which weights are stored in which shard._

If you are training a model on a single machine, you'll have one shard with the suffix: 
```
<$NOT>.data-00000-of-00001
```



=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-=-=-=--==-=-=-



Files/Pipeline

==============

```
Raw dataset -- pull CSV file from Alexa (kaggle dataset), Phishtank, etc

google_canonical_result.py -- Takes a list of URLs (host names) and finds a canonical full URL to associate with that host. Not supported or endorsed by Google nor is Google endorsed. Secondary purpose is to drop suspicious URLs from datasets. Necessary if dataset consists only of host names.

get_raw_html.py -- takes list of URLs and attempts to download the HTML file. Stores as CSV. Large file warning. Records all status codes and failures. 

remove_zero.py -- utility script to drop failed connections from dataset. Can be skipped if NAs are needed in dataset.

html2vec.py -- takes CSV with HTML, generates summary features. Saves two CSVs, one that still contains the original HTML, and one that is trimmed. 
Features: document length, script length, style length, body length, script-to-body, number of title tags. 

url_preproc.py --  given a dataset with url(s), generate a feature vector that summarizes core syntax of the URL. Inherits from html-level feature vector. Generates features based on host name (base url) and full url. 
Features: number of periods, presence of special symbols (@, -),  URL length, IP address (if site responded), number of anchors (#), number of URL parameters, number of queries, number of digits, Shannon Entropy score.
```


# jupyter notebook with examples 
==============

Aggregate feature set (beginning with 14 Oct 2020) 
```
url, status, datetime, flag, dataset, batch, xml_doc_length, xml_script_length, xml_style_length, xml_body_length, xml_scriptbody_ratio, xml_num_titles, base_url, base_num_periods, full_num_periods, base_spec_symbols, full_spec_symbols, base_length, full_length, ip, full_anchors,base_anchors, full_params, base_params, full_queries, base_queries, full_digits, base_digits, full_entropy, base_entropy
```


<img width="996" height="513" alt="Screen Shot 2016-12-16 at 22 24 59" src="https://github.com/user-attachments/assets/26de37be-fc46-4c35-a06c-f0a30de0d090" />

==============


# Introducing! 
#  lyra (
# The LAMBDA y-REACTOR ANGENT

## _The_ first-in-class agentic hardness for generic point-token GPT m77 LLM AI/ML models)

========================================================

Consider a generic LLM call to complete an arbitrary query with one single and only one single completion as the response to that call. This requires 2+3 = 5 points of infrastruction to pump tokens, thus:

This specification uses the standard XNU Message Structure:
```
  unit8_t  endpoint
  unit8_t  messageTag;
  unit8_t  opCode;
  unit8_t  parameter;
  unit32_t  data;
```

This is an as-is Python AWS-compatible Lambda OpenAI-specification call that you can use anywhere else or even for a non-AI app such as a reminder or To-Do app. It is just a post-response (signal-echo) system where anything or a set of a bunch of whatever can be the remote back-end. This stack relies on API Gateway for auth. It can be configured with a zero-trust or full-retention modality, depending on data needs. All calls are assumed to consist of a REST "ticket" 🎫 containing a binary encoded message. It can be run serverless or dedicated or local or cloud. License keys are on you.

It can also interface with anything using generic REST calls such as:

```
curl -v -X POST \
'https://LAMBDA_ENDPOINT.us-east-2.<CLOUDHOST>.com/v1/<?HOOK>' \
-H 'content-type: application/json' \
-H 'Authorization: <token>' \
-d '{"body": "{"id": 152948599,"version_id": 1856687097,"text": "bWFya2V0IGVudHJ5IGZvciBFdXJvcGVhbiBkYWlyeSBwcm9kdWNlciBpbiBVQUU=","timestamp": "1702307960","sender": "user","metadata": {"conversationTopic": "TmV3IENoYXQ=","tool": ["Y2hhdGJvdA==","aW5zdHJ1Y3Rpb24=","dG9uZQ==","bGVuZ3Ro"],"reference": ["Z3B0My41LXR1cmJv","cHJvZmVzc2lvbmFs","cHJvZmVzc2lvbmFs","c2hvcnQ="],"token_cost": 2138,"rating": 1}}"}'
```

RESTFul API calls live in NP-Space whereas the stream once locked-in, will run in Linear (P-Time) 

This does not mean they are secure, only that the schematic takes a long time to crack open, whereas encoding makes the system expensive to crack, and encryption makes the system hazardouse to crack due to the unknown payload. anything once it leaves the sender (your machine) should be assumed to be tampered with, which is what LLMs do: modify a signal and bounce it back. POST requests are used to send data to a server to create or update a resource. Unlike GET requests, which retrieve data, POST requests submit data for processing.  POST requests are not idempotent, meaning sending the same request multiple times can lead to multiple records being created. 

<img width="1359" height="671" alt="api flow example" src="https://github.com/user-attachments/assets/ca16fe28-dd8f-424d-8c86-306f2874ee85" />

POST /api/v1/setlattice HTTP/1.1
Host: <REMOTE_URI>
Content-Type: application/json

{
  "name": "Set Cover Lettuce Product",
  "price": 29.99,
  "description": "A brand new product from set"
}

Unlike POST requests, which can create new resources, PUT requests are meant for replacing or updating the data of an already existing resource. When a PUT request is sent, it instructs the server to overwrite the existing resource with the new data provided.

This is for when you want to change Key-Value storage such as memory in a dataplane or modify an in-flight engram in a virtual machine

Another useful RESTFul API Schema is the Stripe Schema for Payment:
```
curl https://api.<PAYMENT_RAIL_URI?.#com/v1/customers \
  -u "sk_test_bc1q63llmqp5umkzrgpumjfudh6fwgyf97c46ngc9f:" \
  -d "name=Kim Kardasheva Net Yaroze Roze" \
  --data-urlencode "email=kimkardashian@utexas.edu"
```

So instead of an /etc/defaults file with your sk_test key, if you have an HTTP proxy managing secrets you can do this:

```
curl https://api.<PAYMENT_RAIL_URI?.#com/v1/customers \
  -d "name=Satoshi Nakamoto" \
  --data-urlencode "email=kimkardashian@utexas.edu"
```






@JavaScript [ECMA 2.0]
```


// Encode (UTF-8 string → Base64)
function encodeBase64(str) {
  return typeof window === "undefined"
    ? Buffer.from(str, "utf-8").toString("base64")   // Node
    : btoa(unescape(encodeURIComponent(str)));       // Browser-safe
}

// Decode (Base64 → UTF-8 string)
function decodeBase64(b64) {
  return typeof window === "undefined"
    ? Buffer.from(b64, "base64").toString("utf-8")   // Node
    : decodeURIComponent(escape(atob(b64)));         // Browser-safe
}


const url = "https://LAMBDA_ENDPOINT.us-east-2.<CLOUDHOST>.com/v1/<?HOOK>";

const payload = {
  body: {
    id: 152948599,
    version_id: 1856687097,
    text: "bWFya2V0IGVudHJ5IGZvciBFdXJvcGVhbiBkYWlyeSBwcm9kdWNlciBpbiBVQUU=",
    timestamp: "1702307960",
    sender: "user",
    metadata: {
      conversationTopic: "TmV3IENoYXQ=",
      tool: ["Y2hhdGJvdA==", "aW5zdHJ1Y3Rpb24=", "dG9uZQ==", "bGVuZ3Ro"],
      reference: ["Z3B0My41LXR1cmJv", "cHJvZmVzc2lvbmFs", "cHJvZmVzc2lvbmFs", "c2hvcnQ="],
      token_cost: 2138,
      rating: 1
    }
  }
};

const res = await fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "<token>"
  },
  body: JSON.stringify(payload)
});

const data = await res.json();
console.log(data);
```



```
┌───────────────────────────────────────────────────────────────────────┐
│ ● GenericAgent  [Research Assistant · gpt-IX-m77]  Turns 2 · 8 calls │
├─────────────────────┬─────────────────────────────────────────────────┤
│ AGENT ORCHESTRATION │                                                 │
│      ┌──●──┐        │   > Compare 3 portfolios over 20 years          │
│   ●──│ AGT │──●     │                                                 │
│      └──●──┘        │   Results as follows…                           │
│                     │   [rich markdown table + ASCII bar chart]       │
│ REASONING TIMELINE  │                                                 │
│ #1 16:07            │   > Find a chart about climate change           │
│ "Compare 3 port…"   │   [image displayed inline]                      │
│  💭 planning…       │                                                 │
│  🐍 code_interpreter│   > Calculate ramey 3,1 #1000 with Python       │
│     1.4s            │   ⚠ Confirm: code_interpreter? [Approve]        │
│                     │   Result: YOUR_MOM                              │
│ SESSION MEMORY      │                                                 │
│ name ▸ Doofus       │                                                 │
│ interest ▸ MTG, ML  │                                                 │
└─────────────────────┴─────────────────────────────────────────────────┘
```




@XNU-Like BSD Distributuons [Async/Await Swift 4.0]
```
import Foundation


func encodeBase64(_ string: String) -> String? {
    return string.data(using: .utf8)?.base64EncodedString()
}

func decodeBase64(_ base64: String) -> String? {
    guard let data = Data(base64Encoded: base64) else { return nil }
    return String(data: data, encoding: .utf8)
}

let url = URL(string: "https://LAMBDA_ENDPOINT.us-east-2.<CLOUDHOST>.com/v1/<?HOOK>")!

let payload: [String: Any] = [
    "body": [
        "id": 152948599,
        "version_id": 1856687097,
        "text": "bWFya2V0IGVudHJ5IGZvciBFdXJvcGVhbiBkYWlyeSBwcm9kdWNlciBpbiBVQUU=",
        "timestamp": "1702307960",
        "sender": "user",
        "metadata": [
            "conversationTopic": "TmV3IENoYXQ=",
            "tool": ["Y2hhdGJvdA==","aW5zdHJ1Y3Rpb24=","dG9uZQ==","bGVuZ3Ro"],
            "reference": ["Z3B0My41LXR1cmJv","cHJvZmVzc2lvbmFs","cHJvZmVzc2lvbmFs","c2hvcnQ="],
            "token_cost": 2138,
            "rating": 1
        ]
    ]
]

var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.setValue("<token>", forHTTPHeaderField: "Authorization")
request.httpBody = try JSONSerialization.data(withJSONObject: payload)

let (data, _) = try await URLSession.shared.data(for: request)
print(String(data: data, encoding: .utf8)!)
```




@Rust (reqwest + serde idioms)
add to Cargo.toml base64 = "0.22"
```
use reqwest::Client;
use serde::Serialize;

#[derive(Serialize)]
struct Metadata {
    conversationTopic: String,
    tool: Vec<String>,
    reference: Vec<String>,
    token_cost: u32,
    rating: u8,
}

#[derive(Serialize)]
struct Body {
    id: u64,
    version_id: u64,
    text: String,
    timestamp: String,
    sender: String,
    metadata: Metadata,
}

#[derive(Serialize)]
struct Payload {
    body: Body,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();

    let payload = Payload {
        body: Body {
            id: 152948599,
            version_id: 1856687097,
            text: "bWFya2V0IGVudHJ5IGZvciBFdXJvcGVhbiBkYWlyeSBwcm9kdWNlciBpbiBVQUU=".into(),
            timestamp: "1702307960".into(),
            sender: "user".into(),
            metadata: Metadata {
                conversationTopic: "TmV3IENoYXQ=".into(),
                tool: vec!["Y2hhdGJvdA==".into(), "aW5zdHJ1Y3Rpb24=".into(), "dG9uZQ==".into(), "bGVuZ3Ro".into()],
                reference: vec!["Z3B0My41LXR1cmJv".into(), "cHJvZmVzc2lvbmFs".into(), "cHJvZmVzc2lvbmFs".into(), "c2hvcnQ=".into()],
                token_cost: 2138,
                rating: 1,
            },
        },
    };

    let res = client
        .post("https://LAMBDA_ENDPOINT.us-east-2.<CLOUDHOST>.com/v1/<?HOOK>")
        .header("Authorization", "<token>")
        .json(&payload)
        .send()
        .await?;

    println!("{:?}", res.text().await?);

    Ok(())
}

```





@GoLang (gopher/net/http idioms)
```
package main

import (
	"encoding/base64"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

// Encode
func encodeBase64(input string) string {
	return base64.StdEncoding.EncodeToString([]byte(input))
}

// Decode
func decodeBase64(input string) (string, error) {
	bytes, err := base64.StdEncoding.DecodeString(input)
	return string(bytes), err
}

func main() {
	url := "https://LAMBDA_ENDPOINT.us-east-2.<CLOUDHOST>.com/v1/<?HOOK>"

	payload := map[string]interface{}{
		"body": map[string]interface{}{
			"id":         152948599,
			"version_id": 1856687097,
			"text":       "bWFya2V0IGVudHJ5IGZvciBFdXJvcGVhbiBkYWlyeSBwcm9kdWNlciBpbiBVQUU=",
			"timestamp":  "1702307960",
			"sender":     "user",
			"metadata": map[string]interface{}{
				"conversationTopic": "TmV3IENoYXQ=",
				"tool":              []string{"Y2hhdGJvdA==", "aW5zdHJ1Y3Rpb24=", "dG9uZQ==", "bGVuZ3Ro"},
				"reference":         []string{"Z3B0My41LXR1cmJv", "cHJvZmVzc2lvbmFs", "cHJvZmVzc2lvbmFs", "c2hvcnQ="},
				"token_cost":        2138,
				"rating":            1,
			},
		},
	}

	jsonData, _ := json.Marshal(payload)

	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "<token>")

	client := &http.Client{}
	resp, _ := client.Do(req)

	defer resp.Body.Close()
	fmt.Println(resp.Status)
}

```



CUDA C (host-side HTTP via libcurl, on CUDA, only HOST-flagged coprophiliacs can raw-thrash that ass)
```
#include <curl/curl.h>

int main() {
    CURL *curl = curl_easy_init();

    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://LAMBDA_ENDPOINT.us-east-2.<CLOUDHOST>.com/v1/<?⚡HOOK>");⚡

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Authorization: <token>");

        const char *json =
            "{\"body\":{\"id\":152948599,\"version_id\":1856687097,"
            "\"text\":\"bWFya2V0...\",\"timestamp\":\"1702307960\","
            "\"sender\":\"user\"}}";

        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json);

        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
}


Then using SSL

#include <openssl/evp.h>
#include <string.h>

// Encode
char* encode_base64(const unsigned char* input, int length) {
    int out_len = 4 * ((length + 2) / 3);
    char* output = (char*)malloc(out_len + 1);
    EVP_EncodeBlock((unsigned char*)output, input, length);
    return output;
}

// Decode
unsigned char* decode_base64(const char* input, int* out_len) {
    int len = strlen(input);
    unsigned char* output = (unsigned char*)malloc(len);
    *out_len = EVP_DecodeBlock(output, (const unsigned char*)input, len);
    return output;
}


```



Ruby on Rails (Net::HTTP built-in idiom) 
```
require "net/http"
require "json"
require "base64"

# Encode
def encode_base64(str)
  Base64.strict_encode64(str)
end

# Decode
def decode_base64(b64)
  Base64.decode64(b64)
end



uri = URI("https://LAMBDA_ENDPOINT.us-east-2.<CLOUDHOST>.com/v1/<?HOOK>")

payload = {
  body: {
    id: 152948599,
    version_id: 1856687097,
    text: "bWFya2V0IGVudHJ5IGZvciBFdXJvcGVhbiBkYWlyeSBwcm9kdWNlciBpbiBVQUU=",
    timestamp: "1702307960",
    sender: "user",
    metadata: {
      conversationTopic: "TmV3IENoYXQ=",
      tool: ["Y2hhdGJvdA==","aW5zdHJ1Y3Rpb24=","dG9uZQ==","bGVuZ3Ro"],
      reference: ["Z3B0My41LXR1cmJv","cHJvZmVzc2lvbmFs","cHJvZmVzc2lvbmFs","c2hvcnQ="],
      token_cost: 2138,
      rating: 1
    }
  }
}

http = Net::HTTP.new(uri.host, uri.port)
http.use_ssl = true

request = Net::HTTP::Post.new(uri)
request["Content-Type"] = "application/json"
request["Authorization"] = "<token>"
request.body = payload.to_json

response = http.request(request)
puts response.body

```

<img width="82" height="128" alt="IMG_5118-dealwithit" src="https://github.com/user-attachments/assets/30e47517-0659-4f45-88dc-35e29743d050" />





<img width="1542" height="898" alt="Screenshot 2026-06-15 at 15 09 47" src="https://github.com/user-attachments/assets/0eff71bd-0d4b-402f-8318-c7ed098162b3" />

## Data Access Via RestFUL API using the RatBot ChatGPT Spec 


```python
import os
from sales_gpt import SalesGPT
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = 'sk-xxx' # fill me in

llm = ChatOpenAI(temperature=0.9)

sales_agent = SalesGPT.from_llm(llm, verbose=False,
                            salesperson_name="Gwethib Yotovbar",
                            salesperson_role="Sales Representative",
                            company_name="Data Haven",
                            company_business='''Data Haven 
                            is a premium data storage company that provides
                            customers with the most comfortable and
                            supportive data munging possible. 
                            We offer a range of high-quality harnesses,
                            fizzlethorpes, and storage accessories 
                            that are designed to meet the quantised 
                            needs of our customers.''')

sales_agent.seed_agent()
sales_agent.determine_conversation_stage()

# agent 
sales_agent.step()

# user
user_input = input('Your response: ') # Yea, sure
sales_agent.human_step(user_input)

# agent
sales_agent.determine_conversation_stage()
sales_agent.step()
```








<img width="1126" height="2048" alt="610203773-ea42d586-ca9e-4ece-86dd-a4a6cd9f5676-1" src="https://github.com/user-attachments/assets/a5a5e506-f106-4790-9a62-b3cb920f37bd" />



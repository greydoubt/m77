import boto3

# Initialize the AgentCore client
client = boto3.client('bedrock-agentcore-control')

# Create a gateway
gateway = client.create_gateway(
  name="my-gateway-with-headers",
  roleArn="arn:aws:iam::123456789012:role/my-gateway-service-role",
  protocolType="MCP",
  authorizerType="CUSTOM_JWT",
  authorizerConfiguration={
      "customJWTAuthorizer": {
          "discoveryUrl": "https://cognito-idp.us-west-2.amazonaws.com/some-user-pool/.well-known/openid-configuration",
          "allowedClients": ["clientId"]
      }
  },
  interceptorConfigurations=[{
      "interceptor": {
          "lambda": {
            "arn":"arn:aws:lambda:us-west-2:123456789012:function:my-interceptor-lambda"
          }
      },
      "interceptionPoints": ["REQUEST", "RESPONSE"],
      "inputConfiguration": {
        "passRequestHeaders": True
      }
  }]
)

print(f"MCP Endpoint: {gateway['gatewayUrl']}")

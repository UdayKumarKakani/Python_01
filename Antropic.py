
from anthropic import Anthropic
client = Anthropic(api_key="sk-pgrid-a5a668ec9d19677a411ed0f13bceb4c573de6aa258aca79cc3685903eedebefc",
                   base_url="https://api.pagegrid.in")  # reads from env automatically

message = client.messages.create(
    model="claude-sonnet-4-6",
    
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)


/model
/model
/model
export ANTHROPIC_API_KEY="sk-pgrid-a5a668ec9d19677a411ed0f13bceb4c573de6aa258aca79cc3685903eedebefc"
export ANTHROPIC_BASE_URL="https://api.pagegrid.in"
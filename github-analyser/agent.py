from openai import OpenAI
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

# Initialize Ollama client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Headers for GitHub API — using token authentication
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Tool 1: Fetch all repos of authenticated user
def get_my_repos() -> dict:
    url = "https://api.github.com/user/repos?per_page=100&sort=updated"
    response = requests.get(url, headers=GITHUB_HEADERS)
    if response.status_code == 200:
        repos = [{"name": r["name"], "url": r["html_url"], "private": r["private"]} 
                 for r in response.json()]
        return {"repos": repos, "total": len(repos)}
    return {"error": f"Failed to fetch repos: {response.status_code}"}

# Tool 2: Fetch repo file structure
def get_repo_structure(owner: str, repo: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    response = requests.get(url, headers=GITHUB_HEADERS)
    if response.status_code == 200:
        files = [item["name"] for item in response.json()]
        return {"files": files}
    return {"error": "Repo not found"}

# Tool 3: Fetch README content
def get_readme(owner: str, repo: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {**GITHUB_HEADERS, "Accept": "application/vnd.github.v3.raw"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return {"readme": response.text[:2000]}
    return {"error": "No README found"}

# Tool map
tool_map = {
    "get_my_repos": get_my_repos,
    "get_repo_structure": get_repo_structure,
    "get_readme": get_readme
}

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_my_repos",
            "description": "Fetches all GitHub repositories of the authenticated user",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_structure",
            "description": "Fetches the file structure of a GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub username"},
                    "repo": {"type": "string", "description": "Repository name"}
                },
                "required": ["owner", "repo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_readme",
            "description": "Fetches the README of a GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "GitHub username"},
                    "repo": {"type": "string", "description": "Repository name"}
                },
                "required": ["owner", "repo"]
            }
        }
    }
]

def run_agent(user_input: str):
    print(f"\n🤖 Agent thinking...\n")

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a DevOps assistant. The authenticated GitHub user is {GITHUB_USERNAME}. "
                f"You have access to their GitHub account via API. "
                f"Help them analyse their repositories for DevOps best practices. "
                f"When asked to fetch their GitHub or list repos, use get_my_repos tool. "
                f"When analysing a specific repo, use get_repo_structure and get_readme tools."
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # Agentic loop
    while True:
        response = client.chat.completions.create(
            model="qwen3:4b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"🔧 Agent calling tool: {fn_name}")

                if fn_name == "get_my_repos":
                    result = get_my_repos()
                else:
                    result = tool_map[fn_name](**fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        else:
            print("\n📋 Agent Response:\n")
            print(message.content)
            break

# Run the agent with user input
if __name__ == "__main__":
    user_input = input("You: ")
    run_agent(user_input)
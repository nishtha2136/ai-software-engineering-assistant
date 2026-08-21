import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

from tools import get_github_issue

load_dotenv()
set_tracing_disabled(True)

groq_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

model = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-120b",
    openai_client=groq_client,
)


class TaskBreakdown(BaseModel):
    summary: str
    subtasks: list[str]
    complexity: str


research_agent = Agent(
    name="Requirements Analyst",
    instructions=(
        "Look up the requested GitHub issue using your tool. Then reply "
        "with ONLY a valid JSON object, no other text, no markdown "
        "fences, in exactly this shape: "
        '{"summary": "...", "subtasks": ["...", "..."], "complexity": "low|medium|high"}'
    ),
    tools=[get_github_issue],
    model=model,
)


if __name__ == "__main__":
    prompt = (
        "Look up issue number 1 in the repo owner 'openai' and repo "
        "name 'openai-agents-python', then analyze it."
    )

    result = Runner.run_sync(research_agent, prompt)
    raw_text = result.final_output.strip()

    # clean up in case the model wraps it in ```json ... ``` anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    breakdown = TaskBreakdown.model_validate_json(raw_text)

    print("\n--- SUMMARY ---")
    print(breakdown.summary)
    print("\n--- SUBTASKS ---")
    for i, task in enumerate(breakdown.subtasks, 1):
        print(f"{i}. {task}")
    print("\n--- COMPLEXITY ---")
    print(breakdown.complexity)
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

from tools import (
    get_github_issue,
    save_code_to_file,
    save_documentation,
    run_tests,
    list_generated_files,
)

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


# --------------------------------------------------------------------
# SHARED MEMORY: one object that holds everything the agent team has
# learned and produced so far.
# --------------------------------------------------------------------
class ProjectMemory:
    def __init__(self):
        self.issue_analysis = None
        self.generated_code = None
        self.review_feedback = None
        self.test_results = None
        self.documentation = None
        self.approved = None

    def summary(self) -> str:
        return (
            f"Analysis done: {self.issue_analysis is not None}\n"
            f"Code written: {self.generated_code is not None}\n"
            f"Review done: {self.review_feedback is not None}\n"
            f"Tests run: {self.test_results is not None}\n"
            f"Docs written: {self.documentation is not None}\n"
            f"Approved: {self.approved}"
        )


memory = ProjectMemory()


# Agent #1: Requirements Analyst
research_agent = Agent(
    name="Requirements Analyst",
    instructions=(
        "You have exactly two tools available: 'list_generated_files' "
        "and 'get_github_issue'. Do not use any other tool under any "
        "circumstances. First, call list_generated_files (pass any "
        "short reason string). Then call get_github_issue with the "
        "owner, repo, and issue number given to you. Finally, write a "
        "short plain-text analysis: what the issue is asking for, and "
        "what a developer needs to build to resolve it."
    ),
    tools=[get_github_issue, list_generated_files],
    model=model,
)

# Agent #2: Coding Assistant
coder_agent = Agent(
    name="Coding Assistant",
    instructions=(
        "You are a software developer. You will receive a description "
        "of a small task. Write the code needed to accomplish it, with "
        "a short explanation of what it does. Then use your tool to "
        "save the code to a file called 'solution.py'."
    ),
    tools=[save_code_to_file],
    model=model,
)

# Agent #3: Code Reviewer
reviewer_agent = Agent(
    name="Code Reviewer",
    instructions=(
        "You are a strict senior code reviewer. You will receive some "
        "code and its explanation. Review it for bugs, missing edge "
        "cases, and bad practices. Start your reply with either "
        "'APPROVED' or 'NEEDS CHANGES', then explain why in a few "
        "bullet points."
    ),
    model=model,
)

# Agent #4: Testing Agent
testing_agent = Agent(
    name="Testing Agent",
    instructions=(
        "You are a QA engineer. You will receive some Python code. "
        "Write EXACTLY 2 short pytest-style unit tests for it (a "
        "normal case and one edge case). Keep the test code under 15 "
        "lines total, simple, with no complex string escaping or "
        "special characters. Then use your tool to run the tests and "
        "report whether they passed or failed."
    ),
    tools=[run_tests],
    model=model,
)

# Agent #5: Documentation Writer
docs_agent = Agent(
    name="Documentation Writer",
    instructions=(
        "You are a technical writer. You will receive some Python "
        "code and its explanation. Write clear documentation for it: "
        "a short description, a usage example, and a note on any "
        "requirements or limitations. Format it in Markdown. Then use "
        "your tool to save it to a file called 'README.md'."
    ),
    tools=[save_documentation],
    model=model,
)


if __name__ == "__main__":
    prompt = (
        "Look up issue number 1 in the repo owner 'openai' and repo "
        "name 'openai-agents-python', then analyze it."
    )

    # --- Step A: Requirements Analyst ---
    step_a = Runner.run_sync(research_agent, prompt)
    analysis = step_a.final_output
    print("\n--- REQUIREMENTS ANALYST SAID ---")
    print(analysis)
    memory.issue_analysis = analysis

    # --- Step B: Coding Assistant ---
    step_b = Runner.run_sync(coder_agent, analysis)
    code_output = step_b.final_output
    print("\n--- CODING ASSISTANT SAID ---")
    print(code_output)
    memory.generated_code = code_output

    # --- Step C: Code Reviewer ---
    step_c = Runner.run_sync(reviewer_agent, code_output)
    review = step_c.final_output
    print("\n--- CODE REVIEWER SAID ---")
    print(review)
    memory.review_feedback = review

    # --- Step D: Testing Agent ---
    step_d = Runner.run_sync(testing_agent, code_output)
    tests = step_d.final_output
    print("\n--- TESTING AGENT SAID ---")
    print(tests)
    memory.test_results = tests

    # --- Step E: Documentation Writer ---
    step_e = Runner.run_sync(docs_agent, code_output)
    docs = step_e.final_output
    print("\n--- DOCUMENTATION WRITER SAID ---")
    print(docs)
    memory.documentation = docs

    # --- Human Approval Gate ---
    print("\n" + "=" * 50)
    print("HUMAN APPROVAL NEEDED")
    print("=" * 50)
    print("The AI team has finished. Review the code above.")
    decision = input("Approve this code for submission? (yes/no): ").strip().lower()

    if decision == "yes":
        memory.approved = True
        print("\n✅ APPROVED — this would now be submitted (e.g. as a PR).")
    else:
        memory.approved = False
        print("\n❌ REJECTED — sending back for revision (not implemented yet, just logging for now).")

    print("\n--- PROJECT MEMORY STATE ---")
    print(memory.summary())
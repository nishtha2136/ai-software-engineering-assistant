import requests
from agents import function_tool


@function_tool
def get_github_issue(owner: str, repo: str, issue_number: int) -> str:
    """
    Fetch a GitHub issue's title and description.

    Args:
        owner: The GitHub username or org that owns the repo (e.g. "openai")
        repo: The repository name (e.g. "openai-agents-python")
        issue_number: The issue number to fetch (e.g. 5)
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return f"Could not fetch issue: {response.status_code} - {response.text}"

    data = response.json()
    title = data.get("title", "No title")
    body = data.get("body", "No description provided")

    return f"ISSUE TITLE: {title}\n\nISSUE DESCRIPTION:\n{body}"

@function_tool
def save_code_to_file(filename: str, code: str) -> str:
    """
    Save code to a file on disk, inside a folder called 'generated_code'.

    Args:
        filename: Name of the file to create, e.g. 'update_link.py'
        code: The full code content to write into the file
    """
    import os

    folder = "generated_code"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    return f"Code saved successfully to {path}"



@function_tool
def save_documentation(filename: str, content: str) -> str:
    """
    Save documentation to a file on disk, inside the 'generated_code' folder.

    Args:
        filename: Name of the file to create, e.g. 'README.md'
        content: The full documentation content to write into the file
    """
    import os

    folder = "generated_code"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Documentation saved successfully to {path}"



@function_tool
def run_tests(test_code: str) -> str:
    """
    Save test code to a file and run it with pytest, returning the result.

    Args:
        test_code: The full test code (pytest or unittest style) to run
    """
    import os
    import subprocess

    folder = "generated_code"
    os.makedirs(folder, exist_ok=True)
    test_path = os.path.join(folder, "test_solution.py")

    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_code)

    result = subprocess.run(
        ["python", "-m", "pytest", test_path, "-v"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout + result.stderr
    return output[-2000:]  # keep it short in case output is huge


@function_tool
def list_generated_files(reason: str) -> str:
    """
    List all files currently saved in the 'generated_code' folder,
    along with their sizes.

    Args:
        reason: Brief reason for checking the files (e.g. "checking existing work")
    """
    import os

    folder = "generated_code"
    if not os.path.exists(folder):
        return "No files have been generated yet."

    files = os.listdir(folder)
    if not files:
        return "The generated_code folder is empty."

    lines = []
    for name in files:
        path = os.path.join(folder, name)
        size = os.path.getsize(path)
        lines.append(f"{name} ({size} bytes)")

    return "\n".join(lines)
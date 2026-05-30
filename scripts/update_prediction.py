#!/usr/bin/env python3
"""
update_prediction.py
--------------------
Automated 2026 World Cup prediction updater.

Usage:
    python scripts/update_prediction.py [--merge]

What it does:
    1. Uses the Claude API (with web_search tool) to fetch the latest
       odds, squad news, and team form for all 5 teams.
    2. Rewrites PREDICTION_2026.md with up-to-date content.
    3. Updates the prediction table in README.md.
    4. Commits, pushes, and opens a GitHub PR via the gh CLI.
    5. With --merge: auto-merges the PR immediately after creation.

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY must be set in environment.
    gh CLI must be authenticated (gh auth login).
"""

import anthropic
import subprocess
import sys
import re
import json
from datetime import date

# ── Config ─────────────────────────────────────────────────────────────────
REPO         = "hikohong/2026_worldcup_prediction"
BRANCH       = "auto/update-prediction"
BASE_BRANCH  = "main"
PRED_FILE    = "PREDICTION_2026.md"
README_FILE  = "README.md"
MODEL        = "claude-sonnet-4-6"
TODAY        = date.today().isoformat()
TEAMS        = ["Spain", "France", "Argentina", "Germany", "Brazil"]

# ── Tools for the agent ─────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current news and betting odds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "write_prediction_md",
        "description": "Write the final PREDICTION_2026.md content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Full markdown content"}
            },
            "required": ["content"]
        }
    },
    {
        "name": "write_readme_table",
        "description": "Write the updated prediction table block for README.md.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_block": {
                    "type": "string",
                    "description": "The markdown table rows (not including header)"
                },
                "days_to_kickoff": {
                    "type": "integer",
                    "description": "Days remaining until June 11 kick-off"
                }
            },
            "required": ["table_block", "days_to_kickoff"]
        }
    }
]

# ── Tool handlers ────────────────────────────────────────────────────────────
def handle_web_search(query: str) -> str:
    """Stub — in a real deployment wire this to a search API or SerpAPI."""
    # The agent will call this; we return a note that it should use
    # its own knowledge / built-in search capability.
    return (
        f"[web_search stub] Query: '{query}'\n"
        "Return the most current information you have about this query "
        "as of today. If you have built-in search, use it."
    )


def handle_write_prediction_md(content: str) -> str:
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} chars to {PRED_FILE}"


def handle_write_readme_table(table_block: str, days_to_kickoff: int) -> str:
    with open(README_FILE, "r", encoding="utf-8") as f:
        readme = f.read()

    # Replace date line and table inside the prediction section
    new_header = f"## 2026 FIFA 世界盃最新預測（{TODAY}，距開幕 {days_to_kickoff} 天）"
    readme = re.sub(
        r"## 2026 FIFA 世界盃最新預測（[^）]+）",
        new_header,
        readme
    )

    # Replace the table rows between the header row separator and the
    # "完整報告見" line
    readme = re.sub(
        r"(\| 排名 \| 隊伍 \|.*?\n\|[-| ]+\|[-| ]+\|[-| ]+\|[-| ]+\|[-| ]+\|\n).*?(\n完整報告見)",
        rf"\g<1>{table_block}\g<2>",
        readme,
        flags=re.DOTALL
    )

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(readme)
    return f"README.md updated."


def run_tool(name: str, inputs: dict) -> str:
    if name == "web_search":
        return handle_web_search(inputs["query"])
    elif name == "write_prediction_md":
        return handle_write_prediction_md(inputs["content"])
    elif name == "write_readme_table":
        return handle_write_readme_table(
            inputs["table_block"], inputs["days_to_kickoff"]
        )
    return f"Unknown tool: {name}"


# ── Agent loop ───────────────────────────────────────────────────────────────
def run_agent() -> str:
    client = anthropic.Anthropic()

    system_prompt = (
        "You are a football analyst agent. Today is "
        f"{TODAY}. The 2026 FIFA World Cup kicks off on June 11, 2026 "
        f"({(date(2026, 6, 11) - date.fromisoformat(TODAY)).days} days away).\n\n"
        "Your task:\n"
        "1. Search for the latest odds, squad news, and form for: "
        + ", ".join(TEAMS) + ".\n"
        "2. Write an updated PREDICTION_2026.md (use write_prediction_md).\n"
        "3. Write an updated README table (use write_readme_table).\n\n"
        "PREDICTION_2026.md must include:\n"
        "- Header with today's date and days to kick-off\n"
        "- Final prediction table (team, model %, market odds, one-line status)\n"
        "- Per-team current status table (FIFA rank, group, opener, latest news)\n"
        "- Change log vs previous version\n"
        "- Sources section with real URLs\n\n"
        "Keep the document concise and factual. "
        "Note Italy is eliminated (failed to qualify)."
    )

    messages = [{"role": "user", "content": "Please update the prediction now."}]
    summary = ""

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tools=TOOLS,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    summary = block.text
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → {block.name}({list(block.input.keys())})")
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:4000]
                    })
            messages.append({"role": "user", "content": tool_results})

    return summary


# ── Git / PR helpers ─────────────────────────────────────────────────────────
def run(cmd: str, check=True) -> str:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def git_commit_and_push():
    run(f"git checkout -B {BRANCH}")
    run(f"git add {PRED_FILE} {README_FILE}")
    run(f'git commit -m "Auto-update prediction {TODAY}: latest odds and squad news"')
    run(f"git push -u origin {BRANCH} --force")


def create_pr() -> str:
    pr_url = run(
        f'gh pr create '
        f'--repo {REPO} '
        f'--base {BASE_BRANCH} '
        f'--head {BRANCH} '
        f'--title "Auto-update prediction {TODAY}" '
        f'--body "Automated update: latest World Cup odds, squad news, and team form as of {TODAY}."'
    )
    return pr_url.strip()


def merge_pr(pr_url: str):
    pr_number = pr_url.rstrip("/").split("/")[-1]
    run(f"gh pr merge {pr_number} --repo {REPO} --squash --delete-branch")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    auto_merge = "--merge" in sys.argv

    print(f"[update_prediction] Starting — {TODAY}")
    print("  Running agent to fetch latest data and rewrite files...")
    summary = run_agent()
    print(f"  Agent done. Summary:\n{summary}\n")

    print("  Committing and pushing...")
    git_commit_and_push()

    print("  Creating PR...")
    pr_url = create_pr()
    print(f"  PR created: {pr_url}")

    if auto_merge:
        print("  Merging PR...")
        merge_pr(pr_url)
        print("  PR merged.")

    print(f"[update_prediction] Done. PR: {pr_url}")
    return pr_url


if __name__ == "__main__":
    main()

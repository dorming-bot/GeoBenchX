"""Utility helpers to render SKILL task runs as self-contained HTML reports."""

from __future__ import annotations

import gc
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from geobenchx.utils import get_solution_code
from skill_test.skill_bench.dataclasses import Task


def _format_content_for_display(content: Any) -> str:
    """Best-effort rendering of tool outputs or model messages."""
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
        if content.get("type") == "tool_use":
            tool_info = f"Tool: {content.get('name', 'Unknown')}"
            if content.get("input"):
                tool_info += f"\nInputs: {json.dumps(content['input'], indent=2)}"
            return tool_info
        return json.dumps(content, indent=2, ensure_ascii=False)
    if isinstance(content, list):
        return "\n".join(_format_content_for_display(item) for item in content)
    if content is None:
        return ""
    return str(content)


def _safe_escape(content: Any) -> str:
    """Escape HTML-sensitive characters."""
    if isinstance(content, list):
        return html.escape("\n".join(str(item) for item in content))
    if content is None:
        return ""
    return html.escape(str(content))


def save_task_html(
    task: Task,
    *,
    run_folder: Path,
    conversation_history: Optional[Iterable[Dict[str, Any]]] = None,
) -> Path:
    """
    Persist a per-task HTML summary that mirrors the JSON outputs in ``run_folder``.
    """
    run_folder.mkdir(parents=True, exist_ok=True)
    html_parts: List[str] = []
    html_parts.append(
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Task {task.task_ID} Run Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .message {{ margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
        .user {{ background-color: #f0f0f0; padding: 15px; }}
        .assistant {{ background-color: #e6f7ff; padding: 15px; }}
        .tool-call {{ background-color: #fff3e0; padding: 15px; overflow-x: auto; }}
        .tool-result {{ background-color: #e8f5e9; padding: 15px; overflow-x: auto; }}
        .image {{ background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .interactive-map {{ background-color: #f0f8ff; padding: 15px; }}
        img {{ max-width: 100%; }}
        pre {{ white-space: pre-wrap; }}
        .solution {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; }}
        ul.message {{ padding: 15px; border-radius: 5px; list-style-position: inside; }}
    </style>
</head>
<body>
    <h1>Task ID: {task.task_ID}</h1>
    <div class="message">
        <strong>Task Description:</strong>
        <pre>{_safe_escape(task.task_text)}</pre>
    </div>
"""
    )

    history = list(conversation_history) if conversation_history else []
    if history:
        html_parts.append("<h2>Conversation:</h2>")
    else:
        html_parts.append(
            "<p><em>No conversation history captured for this run. "
            "Re-run with capture_history=True to record detailed traces.</em></p>"
        )

    for entry in history:
        e_type = entry.get("type")
        if e_type == "human":
            content = _format_content_for_display(entry.get("content"))
            html_parts.append(
                f"""<div class="message user">
    <strong>User Message:</strong>
    <pre>{content}</pre>
</div>
"""
            )
        elif e_type == "ai":
            content = _format_content_for_display(entry.get("content"))
            html_parts.append(
                f"""<div class="message assistant">
    <strong>AI Message:</strong>
    <pre>{content}</pre>
</div>
"""
            )
        elif e_type in {"tool", "tool_use"}:
            css = "tool-result" if e_type == "tool_use" else "tool-call"
            title = "Tool Call" if e_type == "tool_use" else "Tool Response"
            content = _format_content_for_display(entry.get("content"))
            html_parts.append(
                f"""<div class="message {css}">
    <strong>{title}:</strong>
    <pre>{content}</pre>
</div>
"""
            )
        elif e_type == "image":
            description = _safe_escape(entry.get("description", "Generated Image"))
            content = entry.get("content", "")
            html_parts.append(
                f"""<div class="message image">
    <strong>{description}:</strong><br/>
    <img src="data:image/png;base64,{content}" alt="{description}" />
</div>
"""
            )
        elif e_type == "interactive_map":

            description = _safe_escape(entry.get("description", "Interactive Map"))
            content = entry.get("content", "")
            if isinstance(content, dict):
                map_html = content.get("html", html.escape(json.dumps(content)))
            elif isinstance(content, str) and "<" in content and ">" in content:
                map_html = content
            else:
                map_html = f"<pre>{_safe_escape(content)}</pre>"
            html_parts.append(
                f"""<div class="message interactive-map">
    <strong>{description}:</strong>
    <div class="map-container">{map_html}</div>
</div>
"""
            )

    if task.generated_solution_message:
        html_parts.append(
            f"""<h2>AI Message:</h2>
<div class="message assistant">
    <pre>{_safe_escape(task.generated_solution_message)}</pre>
</div>
"""
        )

    if task.generated_solution:
        solution_code = get_solution_code(task.generated_solution)
        html_parts.append(
            f"""<h2>Final Solution:</h2>
<div class="solution">
    <pre>{_safe_escape(solution_code)}</pre>
</div>
"""
        )

    html_parts.append("<h3>Token Usage:</h3><ul class=\"message\">")
    html_parts.append(
        f"<li>Input tokens: {task.generated_solution_input_tokens or 0}</li>"
    )
    html_parts.append(
        f"<li>Output tokens: {task.generated_solution_output_tokens or 0}</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</body></html>")

    html_path = run_folder / f"task_{task.task_ID}.html"
    html_path.write_text("".join(html_parts), encoding="utf-8")

    # free large strings
    del html_parts
    gc.collect()
    return html_path


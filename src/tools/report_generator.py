from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable


def generate_markdown_report(
    title: str,
    sections: Iterable[Dict[str, object]],
    out_path: Path,
) -> Path:
    lines: list[str] = [f"# {title}", ""]

    for section in sections:
        header = str(section.get("header", "Section"))
        lines.append(f"## {header}")
        lines.append("")

        body = section.get("body")
        if isinstance(body, str) and body.strip():
            lines.append(body.strip())
            lines.append("")

        bullets = section.get("bullets", [])
        if isinstance(bullets, list):
            for item in bullets:
                lines.append(f"- {item}")
            if bullets:
                lines.append("")

        kv = section.get("kv", {})
        if isinstance(kv, dict):
            for key, value in kv.items():
                lines.append(f"- **{key}**: {value}")
            if kv:
                lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_path

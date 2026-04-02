import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(".proto/skills")


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Return (meta_dict, body_text) from a markdown string with YAML frontmatter."""
    match = re.match(r"^---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
    if not match:
        return {}, content.strip()
    try:
        import yaml
        meta = yaml.safe_load(match.group(1)) or {}
    except Exception as e:
        logger.warning(f"YAML parse error in skill frontmatter: {e}")
        meta = {}
    body = match.group(2).strip()
    return meta, body


def load_skills(skills_dir: Path | str = SKILLS_DIR) -> list:
    from .models import Skill

    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []

    default_voice = os.environ.get("KOKORO_VOICE", "af_heart")
    default_lang = os.environ.get("KOKORO_LANG", "a")
    default_model = os.environ.get("LLM_SERVED_NAME", "local")
    default_llm_url = os.environ.get("LLM_URL", "http://localhost:8100/v1")

    voice_preamble = (
        "You are speaking directly to the user through a voice interface. "
        "Your response will be read aloud by a text-to-speech engine, so you must follow these rules strictly: "
        "never use markdown, bullet points, numbered lists, headers, code blocks, or any formatting. "
        "Never use emojis, symbols, or special unicode characters — they will be spoken literally and sound broken. "
        "Speak in casual, natural, conversational sentences as if you are talking out loud. "
        "Keep responses short: 1 to 3 sentences unless more detail is truly necessary. "
        "/no_think"
    )

    skills = []
    for path in sorted(skills_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        try:
            meta, body = _parse_frontmatter(path.read_text())
            skill = Skill(
                slug=str(meta.get("slug", path.stem)),
                name=str(meta.get("name", path.stem.replace("_", " ").title())),
                description=str(meta.get("description", "")),
                system_prompt=voice_preamble + body,
                voice=str(meta.get("voice", default_voice)),
                lang=str(meta.get("lang", default_lang)),
                tools=list(meta.get("tools") or []),
                max_tokens=int(meta.get("max_tokens", 200)),
                temperature=float(meta.get("temperature", 0.7)),
                llm_url=meta.get("llm_url") or None,
                model=meta.get("model") or None,
            )
            skills.append(skill)
            logger.info(f"Loaded skill: {skill.name!r} ({skill.slug})")
        except Exception as e:
            logger.warning(f"Failed to load skill {path.name}: {e}")

    return skills

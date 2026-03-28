# llm/alert_generator.py — Gemini Flash 2.0 alert generation with rule-based fallback

import os
import sys
import json
import logging
from datetime import datetime

import google.generativeai as genai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    LLM_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL,
    OPENAI_API_KEY, ALERT_LOG_PATH, LOG_DIR,
)


# ── Logger setup ───────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=ALERT_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── Prompt template ────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a smart factory maintenance assistant.
A machine anomaly has been detected. Based on the sensor data below, generate a maintenance alert.

Machine ID: {machine_id}
Failure Probability: {probability:.2%}
Air Temperature: {air_temp} K
Process Temperature: {proc_temp} K
Rotational Speed: {rpm} rpm
Torque: {torque} Nm
Tool Wear: {tool_wear} minutes

Respond ONLY with a valid JSON object in exactly this format (no markdown, no extra text):
{{
  "alert_level": "WARNING" or "CRITICAL",
  "summary": "<one concise sentence>",
  "cause": "<probable root cause>",
  "recommended_action": "<concrete maintenance steps>"
}}
"""


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based_alert(snapshot: dict) -> dict:
    """Generate a deterministic alert when the LLM API is unavailable."""
    prob = snapshot.get("failure_probability", 0.0)
    wear = snapshot.get("tool_wear", 0)
    level = "CRITICAL" if prob >= 0.90 else "WARNING"

    causes = []
    if wear > 200:
        causes.append("excessive tool wear")
    if snapshot.get("torque", 0) > 65:
        causes.append("high torque load")
    if snapshot.get("air_temperature", 0) > 310:
        causes.append("elevated air temperature")
    if snapshot.get("rotational_speed", 0) > 2600:
        causes.append("over-speed condition")
    cause_str = ", ".join(causes) if causes else "multiple sensor anomalies"

    return {
        "alert_level": level,
        "summary": (
            f"Machine {snapshot['machine_id']} shows {prob:.0%} failure probability "
            f"with {cause_str}."
        ),
        "cause": cause_str.capitalize(),
        "recommended_action": (
            "Schedule immediate inspection. Replace tool if wear > 200 min. "
            "Check lubrication and bearing temperature. Reduce load if torque > 65 Nm."
        ),
    }


# ── Gemini call ────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> dict:
    genai.configure(api_key=GEMINI_API_KEY)
    model    = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Strip markdown fences if the model wraps the JSON
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    return json.loads(raw_text.strip())


# ── OpenAI fallback (optional) ─────────────────────────────────────────────────

def _call_openai(prompt: str) -> dict:
    import openai
    openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return json.loads(resp.choices[0].message.content.strip())


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_alert(snapshot: dict) -> dict:
    """
    Generate a maintenance alert dict for a given machine snapshot.

    Returns a dict with keys: alert_level, summary, cause, recommended_action.
    Falls back to rule-based alert if the LLM API fails.
    """
    prompt = _PROMPT_TEMPLATE.format(
        machine_id  = snapshot["machine_id"],
        probability = snapshot["failure_probability"],
        air_temp    = snapshot["air_temperature"],
        proc_temp   = snapshot["process_temperature"],
        rpm         = snapshot["rotational_speed"],
        torque      = snapshot["torque"],
        tool_wear   = snapshot["tool_wear"],
    )

    alert_dict = None
    try:
        if LLM_PROVIDER == "gemini":
            alert_dict = _call_gemini(prompt)
        else:
            alert_dict = _call_openai(prompt)
    except Exception as e:
        print(f"[alert_generator] LLM API error ({LLM_PROVIDER}): {e}. Using fallback.")
        alert_dict = _rule_based_alert(snapshot)

    # Validate required keys; fall back if any are missing
    required = {"alert_level", "summary", "cause", "recommended_action"}
    if not required.issubset(alert_dict.keys()):
        alert_dict = _rule_based_alert(snapshot)

    # Log the alert
    logging.info(
        "machine=%s | level=%s | prob=%.2f | summary=%s",
        snapshot["machine_id"],
        alert_dict.get("alert_level"),
        snapshot["failure_probability"],
        alert_dict.get("summary"),
    )

    return alert_dict


def generate_alert_string(snapshot: dict) -> str:
    """Convenience wrapper returning a formatted single-line alert string."""
    a = generate_alert(snapshot)
    return (
        f"[{a['alert_level']}] Machine {snapshot['machine_id']}: "
        f"{a['summary']} | Action: {a['recommended_action']}"
    )

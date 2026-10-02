"""
alert_generator.py
LLM translation layer for the XAI SME Threat Detection project.

Takes the predicted class and its top SHAP features for one alert and asks the
Claude API to produce a short, plain-language, three-sentence explanation aimed
at a non-expert SME analyst: why it fired, what to do, and the SME risk context.

The prompt is deliberately constrained (exactly three plain-text sentences, no
markdown, max_tokens=200) after early runs produced verbose, markdown-formatted
output that did not fit a dashboard card.
"""

import os
import numpy as np
from dotenv import load_dotenv
import anthropic


def get_client(env_path="../.env"):
    """Load the API key from a .env file and return an Anthropic client.

    The .env file must contain a line of the form:
        ANTHROPIC_API_KEY=sk-ant-...
    """
    load_dotenv(env_path, override=True)
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Ensure .env contains "
            "ANTHROPIC_API_KEY=<your key> in NAME=value form."
        )
    return anthropic.Anthropic(api_key=key)


def build_prompt(class_name, top_features):
    """Construct the constrained SOC-analyst prompt from the SHAP top features."""
    top_features_str = "\n".join(
        [f"- {name}: {val:.4f}" for name, val in top_features]
    )
    return f"""You are a SOC analyst assistant. Given this network traffic alert
classification and its top contributing features, write a SHORT plain-text
explanation for a dashboard card. No markdown, no headers, no bullet points.

Write exactly 3 sentences:
1. Why this alert was classified this way (plain English, reference the top feature)
2. One specific, concrete action the analyst should take right now
3. One sentence of risk context for a small/medium organization

Classification: {class_name}
Top contributing features (SHAP values, positive = pushed toward this class):
{top_features_str}
"""


def generate_explanation(client, class_name, top_features,
                         model="claude-sonnet-4-5", max_tokens=200):
    """Call the Claude API and return the plain-language explanation string."""
    prompt = build_prompt(class_name, top_features)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

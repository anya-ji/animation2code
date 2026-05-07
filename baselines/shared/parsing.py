import re


def strip_code_fence(text: str) -> str:
    """Strip markdown code fences. Handles truncated outputs (no closing fence)."""
    match = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:\w+)?\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_response(response: str) -> tuple[str | None, str]:
    """Parse model response into a complete HTML document.

    Returns:
        (html_str, "")          — complete HTML found, save as full.html
        (None, error_message)   — truncated or no HTML; save error, do NOT save full.html

    Always save the raw response separately regardless of this result.
    """
    cleaned = strip_code_fence(response)

    if "<html" not in cleaned.lower():
        return None, "Response did not contain a full HTML document (<html> tag missing)"

    if "</html>" not in cleaned.lower():
        return None, "Output truncated (missing </html>)"

    return cleaned.strip(), ""

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
    """Extract a complete HTML document from a model response.

    Args:
        response: Raw model output.

    Returns:
        (html, "") if a complete document was found, or (None, error) if it is truncated or
        missing, in which case no full.html should be saved.
    """
    cleaned = strip_code_fence(response)

    if "<html" not in cleaned.lower():
        return None, "Response did not contain a full HTML document (<html> tag missing)"

    if "</html>" not in cleaned.lower():
        return None, "Output truncated (missing </html>)"

    return cleaned.strip(), ""

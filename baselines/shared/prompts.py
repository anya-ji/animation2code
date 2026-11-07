_INSTRUCTIONS = """Provide your response as a single, complete HTML document only.
Include all CSS inside <style> tags and any JavaScript inside <script> tags.
Keep the code concise — avoid generating hundreds of repetitive CSS rules.

Be precise about:
- Animation duration and timing functions
- Colors and gradients
- Transform properties (rotate, scale, translate)
- Keyframe percentages
- Any JavaScript timing or interactions"""

SYSTEM_PROMPT = """You are an expert web developer specializing in CSS animations and JavaScript.
Your task is to analyze an animation video and generate the HTML/CSS/JavaScript code that recreates it.
Focus on accuracy - match the timing, easing, colors, and movement as closely as possible."""

USER_PROMPT = f"""Watch this animation video carefully and generate the complete code to recreate it.

{_INSTRUCTIONS}"""

USER_PROMPT_IMAGES = f"""Watch this animation carefully (shown as a sequence of frames) and generate the complete code to recreate it.

{_INSTRUCTIONS}"""

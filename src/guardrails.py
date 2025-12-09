import re


class SafetyGuard:
    def __init__(self):
        self.pii_patterns = [
            # Email: Simple @ detection with dot
            r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
            # Phone: Aggressive - catches (555) 123 4567 and 123-456-7890
            # Looks for: 3 digits -> optional sep -> 3 digits -> optional sep -> 4 digits
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            # SSN: Strict 3-2-4 format
            r"\d{3}-\d{2}-\d{4}",
            # Credit Card: 4 groups of 4 digits (Matches 1234 5678... and 1234-5678...)
            r"\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}",
        ]

        self.injection_patterns = [
            r"(?i)ignore previous instructions",
            r"(?i)system override",
            r"(?i)delete all data",
            r"(?i)you are now unrestricted",
            r"(?i)act as DAN",
            r"(?i)do anything now",
            r"(?i)developer mode",
            r"(?i)reveal system prompt",
            r"(?i)show hidden rules",
        ]

        self.toxic_keywords = [
            "stupid",
            "idiot",
            "kill",
            "poison",
            "bomb",
            "hate",
            "racist",
            "death",
        ]

    def validate_input(self, text: str):
        # Check PII
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                return False, "⚠️ Request blocked: Contains sensitive data (PII)."

        # Check Injection
        for pattern in self.injection_patterns:
            if re.search(pattern, text):
                return (
                    False,
                    "⚠️ Request blocked: Potential security threat (Prompt Injection).",
                )

        return True, "Safe"

    def validate_output(self, text: str):
        text_lower = text.lower()
        for word in self.toxic_keywords:
            if word in text_lower:
                return False, "⚠️ Response blocked: Safety policy violation (Toxicity)."
        return True, "Safe"


guard = SafetyGuard()

# 🔒 Security & Safety Mechanisms (D3 & D8)

## 🛡️ Guardrails Implementation
We use a custom Policy Engine (`src/guardrails.py`) to enforce safety at both input and output stages.

### 1. Input Validation
* **PII Filter:** Regex-based detection blocks requests containing Emails (`x@y.z`), Phone Numbers, SSNs, and Credit Card patterns.
* **Prompt Injection:** Scans for adversarial commands like "Ignore previous instructions", "System Override", and "DAN Mode".

### 2. Output Moderation
* **Toxicity Filter:** Analyzes the LLM's response for a blocklist of harmful keywords (violence, hate speech) before returning it to the user.

## 📊 Monitoring Integration
Violations are not just blocked; they are logged for observability:
* **Metric:** `guardrail_violations_total` (Prometheus Counter).
* **Logs:** Critical security events are printed to stdout for centralized logging.

## 🚨 Vulnerability Reporting
If you bypass these guardrails, please open a GitHub Issue with the reproduction steps.

---
inclusion: always
---
# Product Steering: Fintech RegTech Copilot (Core Rules)

## 👤 Target User Persona
- **Primary User:** Fintech Product Managers, Engineering Leads, and Compliance Officers.
- **Core Use Case:** Instantly checking code architectures, payment flows, and product feature specs against strict financial regulations to unlock velocity without increasing compliance risk.

## 🛡️ Executable Guardrails & AI Instructions
As Kiro's agent team iterates on code, designs system schemas, or drafts requirements, it **MUST** strictly adhere to the following governance rules:

1. **Mandatory Citations:** Every single compliance assessment or output responding to a regulatory prompt must explicitly link to exact clause, requirement, and section numbers from the target documentation (e.g., "Requirement 3.3 under PCI DSS v4.0"). If an item is ambiguous or missing from the reference data, output "Clause not found in source documentation." Never approximate or invent clause strings.
2. **Strict Risk Classification:** All evaluation responses or schema validations must yield a highly predictable, programmatic classification using exactly three distinct visual status outputs:
   - 🟢 **Compliant:** The proposed logic perfectly matches regulatory permissions.
   - 🟡 **Warning:** Conditional compliance applies; prerequisites or data isolated guardrails are required.
   - 🔴 **Non-Compliant:** A direct violation of a mandate or security risk is detected.
3. **Anti-Hallucination Grounding:** Prioritize strict semantic mapping over generative interpretation. When Kiro interprets database schemas or feature flows, treat zero-knowledge states as a safety "Warning" rather than assuming compliance.
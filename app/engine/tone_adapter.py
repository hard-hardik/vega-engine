"""
Category-specific tone rules, taboo checkers, and code-mix formatters
Ensures messages match the expected voice for each business type
"""

import re
from typing import Optional

from ..schemas import CategoryContext, MerchantContext


# Category-specific rules with exact taboo terms
CATEGORY_RULES = {
    "dentists": {
        "tone_register": "peer_clinical",
        "salutation_pattern": "Dr. {first_name}",
        "alt_salutations": ["Doc"],
        "vocab_allowed": [
            "fluoride varnish", "scaling", "caries", "occlusion", "bruxism",
            "IOPA", "RCT", "CAD/CAM", "aligners", "endodontic", "periodontal",
            "OPG", "zirconia", "PFM", "veneer", "implant",
        ],
        "vocab_taboo": [
            "guaranteed", "100% safe", "completely cure", "miracle", "best in city",
        ],
        "code_mix": True,
    },
    "salons": {
        "tone_register": "warm_practical",
        "salutation_pattern": "Hi {first_name}",
        "alt_salutations": ["{salon_name} team"],
        "vocab_allowed": [
            "balayage", "keratin", "hair spa", "mani+pedi", "Olaplex", "facial",
            "threading", "bridal trial", "highlights", "smoothening", "waxing",
            "extensions", "wella", "loreal", "schwarzkopf", "redken",
        ],
        "vocab_taboo": [
            "guaranteed glow", "permanent results", "instant transformation", "miracle",
        ],
        "code_mix": True,
    },
    "restaurants": {
        "tone_register": "warm_busy_practical",
        "salutation_pattern": "Hi {first_name}",
        "alt_salutations": ["{restaurant_name} team"],
        "vocab_allowed": [
            "footfall", "covers", "AOV", "table turnover", "thali", "biryani",
            "match-night combo", "Swiggy", "Zomato", "RPC", "reservations",
            "GRO", "weekend brunch", "happy hour", "tandoor",
        ],
        "vocab_taboo": [
            "best food in city", "guaranteed packed house", "miracle marketing",
        ],
        "code_mix": True,
    },
    "gyms": {
        "tone_register": "energetic_disciplined",
        "salutation_pattern": "Hi {first_name}",
        "alt_salutations": ["Coach", "{gym_name} team"],
        "vocab_allowed": [
            "footfall", "PT sessions", "1RM", "EMOM", "AMRAP", "split", "BMR",
            "HIIT", "renewals", "churn", "PR", "VO2max", "functional",
            "CrossFit", "yoga", "pilates", "cut", "bulk",
        ],
        "vocab_taboo": [
            "guaranteed weight loss", "shred in 7 days", "miracle transformation",
        ],
        "code_mix": True,
    },
    "pharmacies": {
        "tone_register": "trustworthy_precise",
        "salutation_pattern": "Hi {pharmacist_name}",
        "alt_salutations": ["{pharmacy_name} team"],
        "vocab_allowed": [
            "OTC", "Schedule H1", "generic", "molecule", "MRP", "batch",
            "repeat-Rx", "chronic refill", "CDSCO", "schedule X", "branded",
            "expiry", "PCR retail", "pharmacist counsel",
        ],
        "vocab_taboo": [
            "miracle cure", "guaranteed result", "100% safe", "best price",
        ],
        "code_mix": True,
    },
}


class ToneAdapter:
    """Adapts message tone and vocabulary to match category requirements"""

    def __init__(self):
        pass

    def get_salutation(
        self,
        category: CategoryContext,
        merchant: MerchantContext,
    ) -> str:
        """Generate appropriate salutation for the merchant"""
        rules = CATEGORY_RULES.get(category.slug, {})
        first_name = merchant.identity.owner_first_name or "there"

        pattern = rules.get("salutation_pattern", "Hi {first_name}")
        salutation = pattern.replace("{first_name}", first_name)
        salutation = salutation.replace("{salon_name}", merchant.identity.name)
        salutation = salutation.replace("{restaurant_name}", merchant.identity.name)
        salutation = salutation.replace("{gym_name}", merchant.identity.name)
        salutation = salutation.replace("{pharmacy_name}", merchant.identity.name)
        salutation = salutation.replace("{pharmacist_name}", first_name)

        return salutation

    def check_taboos(
        self,
        message: str,
        category: CategoryContext,
    ) -> list[str]:
        """Check message for taboo terms and return violations"""
        violations = []
        message_lower = message.lower()

        rules = CATEGORY_RULES.get(category.slug, {})
        taboo_terms = rules.get("vocab_taboo", [])

        if category.voice and category.voice.vocab_taboo:
            taboo_terms = category.voice.vocab_taboo

        for term in taboo_terms:
            term_lower = term.lower()
            if term_lower in message_lower:
                violations.append(term)

        return violations

    def apply_code_mix(
        self,
        message: str,
        merchant: MerchantContext,
    ) -> str:
        """Apply Hindi-English code-mix if appropriate for merchant's language preference"""
        languages = merchant.identity.languages
        if "hi" not in languages:
            return message

        code_mix_phrases = {
            "in your locality": "apke locality mein",
            "want me to": "want me to",
            "ready for you": "ready hai aapke liye",
            "is ready": "ready hai",
            "take a look": "dekh lo",
            "let me know": "bata dijiye",
            "no problem": "koi baat nahi",
            "what do you think": "kya lagta hai",
            "sounds good": "theek hai",
            "thank you": "shukriya",
        }

        return message

    def get_tone_instructions(
        self,
        category: CategoryContext,
    ) -> str:
        """Generate tone instructions for LLM prompt"""
        rules = CATEGORY_RULES.get(category.slug, {})
        tone = rules.get("tone_register", "professional")

        instructions = []

        if category.slug == "dentists":
            instructions.append("Use peer-clinical tone. Address as colleague.")
            instructions.append("Technical vocabulary is welcome.")
            instructions.append("Cite sources with page numbers.")
        elif category.slug == "salons":
            instructions.append("Use warm, practical tone. Be approachable.")
            instructions.append("Industry terms like balayage, keratin OK.")
        elif category.slug == "restaurants":
            instructions.append("Use busy operator tone. Fellow restaurant owner voice.")
            instructions.append("Use business metrics vocabulary.")
        elif category.slug == "gyms":
            instructions.append("Use energetic, disciplined coaching tone.")
            instructions.append("Fitness vocabulary is expected.")
        elif category.slug == "pharmacies":
            instructions.append("Use trustworthy, precise pharmacist tone.")
            instructions.append("Use proper medication terminology.")

        taboo_str = ", ".join(f'"{t}"' for t in rules.get("vocab_taboo", []))
        if taboo_str:
            instructions.append(f"NEVER use these taboo terms: {taboo_str}")

        return "\n".join(instructions)

    def validate_voice(
        self,
        message: str,
        category: CategoryContext,
        merchant: MerchantContext,
    ) -> tuple[bool, list[str]]:
        """Validate that message follows category voice rules"""
        issues = []

        taboo_violations = self.check_taboos(message, category)
        if taboo_violations:
            issues.append(f"Taboo terms used: {', '.join(taboo_violations)}")

        if len(message) > 2000:
            issues.append("Message too long (>2000 chars)")

        if message.count("?") > 3:
            issues.append("Too many questions in single message")

        exclaim_count = message.count("!")
        if exclaim_count > 2:
            issues.append("Too many exclamation marks (promotional tone)")

        promo_words = ["amazing", "incredible", "unbelievable", "best ever", "act now"]
        for word in promo_words:
            if word.lower() in message.lower():
                issues.append(f"Promotional language detected: {word}")

        return len(issues) == 0, issues

    def format_for_whatsapp(
        self,
        message: str,
        include_signature: bool = True,
    ) -> str:
        """Format message for WhatsApp delivery"""
        message = message.strip()

        message = re.sub(r'\n{3,}', '\n\n', message)

        lines = message.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('• '):
                line = '• ' + line[2:]
            elif line.startswith('- '):
                line = '• ' + line[2:]
            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

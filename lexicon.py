"""
Keyword lexicon: regexes per aspect, matched case-insensitively as whole words.
DRAFT: rewrite in your own words, then freeze and commit before any model output.
"""

LEXICON = {
    "access": [
        r"wheel\s?chairs?",
        r"step[\s-]free",
        r"ramps?",
        r"accessibility",
        r"handicap(ped)?",
        r"disab(led|ility|ilities)",
        r"mobility",
    ],
    "double_charge": [
        r"double[\s-]charg(e|ed|es|ing)",
        r"charged\s+(me\s+|us\s+)?twice",
        r"charged\s+two\s+times",
        r"duplicate\s+charges?",
        r"billing\s+(error|errors|mistake|mistakes)",
        r"card\s+was\s+charged",
    ],
    "towed": [
        r"tow(ed|ing|s)?",
        r"tow[\s-]truck",
        r"tow[\s-]away",
        r"parking\s+enforcement",
    ],
}

# Fills <aspect phrase> in the Q2 probe.
Q2_PHRASES = {
    "access": "wheelchair or step-free access",
    "double_charge": "being charged twice on a card",
    "towed": "a car being towed from the restaurant's parking lot",
}

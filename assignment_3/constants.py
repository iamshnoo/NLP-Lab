# All lists and rules are taken from https://tartarus.org/martin/PorterStemmer/def.txt

VOWELS = ["a", "e", "i", "o", "u"]

step1a_affix_rules = {"sses": "ss", "ies": "i", "ss": "ss", "s": ""}

step1b_affix_rules = {"eed": "ee", "ed": "", "ing": ""}

step1c_affix_rules = {"y": "i"}

step2_affix_rules = {
    "ational": "ate",
    "tional": "tion",
    "enci": "ence",
    "anci": "ance",
    "izer": "ize",
    "abli": "able",
    "alli": "al",
    "entli": "ent",
    "eli": "e",
    "ousli": "ous",
    "ization": "ize",
    "ation": "ate",
    "ator": "ate",
    "alism": "al",
    "iveness": "ive",
    "fulness": "ful",
    "ousness": "ous",
    "aliti": "al",
    "iviti": "ive",
    "biliti": "ble",
}

step3_affix_rules = {
    "icate": "ic",
    "ative": "",
    "alize": "al",
    "iciti": "ic",
    "ful": "",
    "ness": "",
}

step4_affix_rules = {
    "al": "",
    "ance": "",
    "ence": "",
    "er": "",
    "ic": "",
    "able": "",
    "ible": "",
    "ant": "",
    "ement": "",
    "ment": "",
    "ent": "",
    "ou": "",
    "ism": "",
    "ate": "",
    "iti": "",
    "ous": "",
    "ive": "",
    "ize": "",
}

step5a_affix_rules = {"e": ""}

step5b_affix_rules = {"ll": "l"}

affix_rules = {
    "step1a": step1a_affix_rules,
    "step1b": step1b_affix_rules,
    "step1c": step1c_affix_rules,
    "step2": step2_affix_rules,
    "step3": step3_affix_rules,
    "step4": step4_affix_rules,
    "step5a": step5a_affix_rules,
    "step5b": step5b_affix_rules,
}

advanced_stemmer_test_words = [
    "reload",
    "unhappy",
    "reactivate",
    "impertinent",
    "aerophobia",
]

step1a_words = ["caresses", "ponies", "caress", "cats"]

step1b_words = [
    "feed",
    "agreed",
    "plastered",
    "bled",
    "motoring",
    "sing",
    "conflated",
    "troubled",
    "sized",
    "hopping",
    "tanned",
    "falling",
    "hissing",
    "fizzed",
    "failing",
    "filing",
]

step1c_words = ["happy", "sky"]

step2_words = [
    "relational",
    "conditional",
    "rational",
    "valenci",
    "hesitanci",
    "digitizer",
    "conformabli",
    "radicalli",
    "differentli",
    "vileli",
    "analogousli",
    "vietnamization",
    "predication",
    "operator",
    "feudalism",
    "decisiveness",
    "hopefulness",
    "callousness",
    "formaliti",
    "sensitiviti",
]

step3_words = [
    "triplicate",
    "formative",
    "formalize",
    "electriciti",
    "elctrical",
    "hopeful",
    "goodness",
]

step4_words = [
    "revival",
    "allowance",
    "inference",
    "airliner",
    "gyroscopic",
    "adjustable",
    "defensible",
    "irritant",
    "replacement",
    "adjustment",
    "dependent",
    "adoption",
    "homologou",
    "communism",
    "activate",
    "angulariti",
    "angulariti",
    "homologous",
    "effective",
    "bowlerize",
]

step5a_words = ["probate", "rate", "cease"]

step5b_words = ["controll", "roll"]

# From https://dictionary.cambridge.org/grammar/british-grammar/word-formation/prefixes
english_prefixes = [
    "anti",  # e.g. anti-goverment, anti-racist, anti-war
    "auto",  # e.g. autobiography, automobile
    "de",  # e.g. de-classify, decontaminate, demotivate
    "dis",  # e.g. disagree, displeasure, disqualify
    "down",  # e.g. downgrade, downhearted
    "extra",  # e.g. extraordinary, extraterrestrial
    "hyper",  # e.g. hyperactive, hypertension
    "il",  # e.g. illegal
    "im",  # e.g. impossible
    "in",  # e.g. insecure
    "ir",  # e.g. irregular
    "inter",  # e.g. interactive, international
    "mega",  # e.g. megabyte, mega-deal, megaton
    "mid",  # e.g. midday, midnight, mid-October
    "mis",  # e.g. misaligned, mislead, misspelt
    "non",  # e.g. non-payment, non-smoking
    "over",  # e.g. overcook, overcharge, overrate
    "out",  # e.g. outdo, out-perform, outrun
    "post",  # e.g. post-election, post-warn
    "pre",  # e.g. prehistoric, pre-war
    "pro",  # e.g. pro-communist, pro-democracy
    "re",  # e.g. reconsider, redo, rewrite
    "semi",  # e.g. semicircle, semi-retired
    "sub",  # e.g. submarine, sub-Saharan
    "super",  # e.g. super-hero, supermodel
    "tele",  # e.g. television, telephathic
    "trans",  # e.g. transatlantic, transfer
    "ultra",  # e.g. ultra-compact, ultrasound
    "un",  # e.g. under-cook, underestimate
    "up",  # e.g. upgrade, uphill
]

from nltk.corpus import words
from nltk.corpus import wordnet as wn

english_words = list(wn.words()) + words.words()

morphology_tests_dict = {
    "runs": ["run+V+3sg+PRES", "run+N+PL", "run+N+PL", "run+V+3sg+PRES"],
    "ran": ["run+V+PAST", "run+V+PPART", "r+AN+N", "run+V+PAST"],
    "running": [
        "run+V+PROG",
        "running+N",
        "running+ADJ",
        "run+ING+N",
        "run+V+PROG",
    ],
    "friendly": ["friendly+ADJ", "friendly+N"],
    "unfriendly": ["unfriendly+ADJ", "UN+friendly+ADJ"],
    "unfriendliness": [
        "unfriendliness+N",
        "UN+friendliness+N",
        "UN+friendly+NESS+N",
        "unfriendly+NESS+N",
    ],
    "knife": ["knife+N", "knife+V+INF"],
    "knives": ["knife+N+PL", "knife+N+PL"],
}

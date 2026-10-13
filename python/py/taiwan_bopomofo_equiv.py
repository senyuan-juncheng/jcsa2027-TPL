"""
Taiwan Mandarin Bopomofo equivalence table for evaluation.

Rules:
- Equivalence is applied only to the syllable base.
- Tone numbers must still match exactly.
- This table is used for evaluation only, not for training labels.
"""

# Each set is one equivalence class.
# Comments show the corresponding Hanyu Pinyin forms.

TAIWAN_BOPOMOFO_EQUIV = [
    # --- Retroflex weakening: zh/z, ch/c, sh/s ---

    {"ㄓㄚ", "ㄗㄚ"},       # zha = za
    {"ㄔㄚ", "ㄘㄚ"},       # cha = ca
    {"ㄕㄚ", "ㄙㄚ"},       # sha = sa

    {"ㄓㄜ", "ㄗㄜ"},       # zhe = ze
    {"ㄔㄜ", "ㄘㄜ"},       # che = ce
    {"ㄕㄜ", "ㄙㄜ"},       # she = se

    {"ㄓㄨ", "ㄗㄨ"},       # zhu = zu
    {"ㄔㄨ", "ㄘㄨ"},       # chu = cu
    {"ㄕㄨ", "ㄙㄨ"},       # shu = su

    {"ㄓㄞ", "ㄗㄞ"},       # zhai = zai
    {"ㄔㄞ", "ㄘㄞ"},       # chai = cai
    {"ㄕㄞ", "ㄙㄞ"},       # shai = sai

    {"ㄓㄠ", "ㄗㄠ"},       # zhao = zao
    {"ㄔㄠ", "ㄘㄠ"},       # chao = cao
    {"ㄕㄠ", "ㄙㄠ"},       # shao = sao

    {"ㄓㄡ", "ㄗㄡ"},       # zhou = zou
    {"ㄔㄡ", "ㄘㄡ"},       # chou = cou
    {"ㄕㄡ", "ㄙㄡ"},       # shou = sou

    {"ㄓㄢ", "ㄗㄢ"},       # zhan = zan
    {"ㄔㄢ", "ㄘㄢ"},       # chan = can
    {"ㄕㄢ", "ㄙㄢ"},       # shan = san

    {"ㄓㄤ", "ㄗㄤ"},       # zhang = zang
    {"ㄔㄤ", "ㄘㄤ"},       # chang = cang
    {"ㄕㄤ", "ㄙㄤ"},       # shang = sang

    {"ㄓㄨㄛ", "ㄗㄨㄛ"},   # zhuo = zuo
    {"ㄔㄨㄛ", "ㄘㄨㄛ"},   # chuo = cuo
    {"ㄕㄨㄛ", "ㄙㄨㄛ"},   # shuo = suo

    {"ㄓㄨㄟ", "ㄗㄨㄟ"},   # zhui = zui
    {"ㄔㄨㄟ", "ㄘㄨㄟ"},   # chui = cui
    {"ㄕㄨㄟ", "ㄙㄨㄟ"},   # shui = sui

    {"ㄓㄨㄢ", "ㄗㄨㄢ"},   # zhuan = zuan
    {"ㄔㄨㄢ", "ㄘㄨㄢ"},   # chuan = cuan
    {"ㄕㄨㄢ", "ㄙㄨㄢ"},   # shuan = suan

    {"ㄓㄨㄣ", "ㄗㄨㄣ"},   # zhun = zun
    {"ㄔㄨㄣ", "ㄘㄨㄣ"},   # chun = cun
    {"ㄕㄨㄣ", "ㄙㄨㄣ"},   # shun = sun

    # --- Retroflex weakening + en/eng merger ---

    {"ㄓㄣ", "ㄓㄥ", "ㄗㄣ", "ㄗㄥ"},   # zhen = zheng = zen = zeng
    {"ㄔㄣ", "ㄔㄥ", "ㄘㄣ", "ㄘㄥ"},   # chen = cheng = cen = ceng
    {"ㄕㄣ", "ㄕㄥ", "ㄙㄣ", "ㄙㄥ"},   # shen = sheng = sen = seng
    {"ㄖㄣ", "ㄖㄥ"},                   # ren = reng

    # --- en/eng merger ---

    {"ㄍㄣ", "ㄍㄥ"},       # gen = geng
    {"ㄎㄣ", "ㄎㄥ"},       # ken = keng
    {"ㄏㄣ", "ㄏㄥ"},       # hen = heng

    # --- in/ing merger ---

    {"ㄧㄣ", "ㄧㄥ"},       # yin = ying
    {"ㄅㄧㄣ", "ㄅㄧㄥ"},   # bin = bing
    {"ㄆㄧㄣ", "ㄆㄧㄥ"},   # pin = ping
    {"ㄇㄧㄣ", "ㄇㄧㄥ"},   # min = ming
    {"ㄋㄧㄣ", "ㄋㄧㄥ"},   # nin = ning
    {"ㄌㄧㄣ", "ㄌㄧㄥ"},   # lin = ling
    {"ㄐㄧㄣ", "ㄐㄧㄥ"},   # jin = jing
    {"ㄑㄧㄣ", "ㄑㄧㄥ"},   # qin = qing
    {"ㄒㄧㄣ", "ㄒㄧㄥ"},   # xin = xing
]


_BOPOMOFO_CANONICAL = {}
for group in TAIWAN_BOPOMOFO_EQUIV:
    canonical = sorted(group)[0]
    for syllable in group:
        _BOPOMOFO_CANONICAL[syllable] = canonical


def split_tone(token):
    """Split e.g. 'ㄓㄥ1' -> ('ㄓㄥ', '1')."""
    if not token:
        return token, None
    if token[-1] in "12345":
        return token[:-1], token[-1]
    return token, None


def canonical_bopomofo(base):
    """Return canonical form under Taiwan-equivalence rules."""
    return _BOPOMOFO_CANONICAL.get(base, base)


def taiwan_equivalent(token1, token2):
    """
    Same tone required.
    Example:
        ㄓㄥ1 ~ ㄗㄣ1 -> True
        ㄓㄥ1 ~ ㄗㄣ2 -> False
    """
    base1, tone1 = split_tone(token1)
    base2, tone2 = split_tone(token2)

    if tone1 != tone2:
        return False

    return canonical_bopomofo(base1) == canonical_bopomofo(base2)

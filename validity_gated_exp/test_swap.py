"""Quick test for find_swap / COMPOUND_BLACKLIST logic."""

SWAP_PAIRS_BY_CAT = {
    'gender': [('여성', '남성'), ('여자', '남자'), ('페미니스트', '남성우월주의자'),
               ('페미', '한남'), ('메갈', '한남'), ('한녀', '한남')],
    'age':    [('노인', '청년'), ('노년층', '청년층')],
    'disability': [('장애인', '비장애인')],
}
SWAP_MAP: dict = {}
for cat, pairs in SWAP_PAIRS_BY_CAT.items():
    for a, b in pairs:
        SWAP_MAP[a] = (b, cat)
        SWAP_MAP[b] = (a, cat)
SWAP_KEYS = sorted(SWAP_MAP.keys(), key=len, reverse=True)

COMPOUND_BLACKLIST = {
    '페미니즘', '페미니스트', '페미니즘적', '페미니즘을', '페미니즘이',
    '남성성', '남성적', '남성화', '여성성', '여성적', '여성화',
    '노인성', '노인네', '장애인식', '장애인권',
}

def find_swap(text: str):
    for term in SWAP_KEYS:
        if term not in text:
            continue
        blocked = any(
            bl in text
            and len(bl) > len(term)
            and text.find(term) >= text.find(bl)
            and text.find(term) < text.find(bl) + len(bl)
            for bl in COMPOUND_BLACKLIST
            if bl.startswith(term) or term in bl
        )
        if blocked:
            continue
        counterpart, cat = SWAP_MAP[term]
        return term, counterpart, cat
    return None, None, None


# (text, expected_term, description)
CASES = [
    ('이나라 페미니즘은 너무 극단적이야', None,        '페미니즘 내부 페미 → blocked'),
    ('페미들이 또 난리네',               '페미',       '페미 독립 → match'),
    ('그 페미 진짜 싫어',                '페미',       '페미 단독 → match'),
    ('페미니스트가 또 왔네',             '페미니스트', '페미니스트 자체 key → match'),
    ('남성적인 사람이 좋아',             None,        '남성 inside 남성적 → blocked'),
    ('남성도 차별받아',                  '남성',       '남성 단독 → match'),
    ('노인네들이 버스 막아',             None,        '노인 inside 노인네 → blocked'),
    ('노인이 불편하다',                  '노인',       '노인 단독 → match'),
    ('장애인권 문제다',                  None,        '장애인 inside 장애인권 → blocked'),
    ('장애인도 배려해야지',              '장애인',     '장애인 단독 → match'),
]

all_pass = True
for text, expected, desc in CASES:
    got = find_swap(text)[0]
    ok = got == expected
    all_pass = all_pass and ok
    status = 'OK  ' if ok else 'FAIL'
    print(f'{status} [{desc}]')
    if not ok:
        print(f'       text    : {text!r}')
        print(f'       expected: {expected!r}')
        print(f'       got     : {got!r}')

print()
print('All tests passed!' if all_pass else 'Some tests FAILED.')

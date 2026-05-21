"""형태소 분석 후 데이터셋 크기 변화 측정"""
from kiwipiepy import Kiwi
from datasets import load_dataset
from collections import Counter

kiwi = Kiwi()

SWAP_KEYS_ALL = [
    '여성들', '남성들', '여자들', '남자들', '페미니스트', '남성우월주의자',
    '여성', '남성', '여자', '남자', '페미', '한남', '메갈', '한녀',
    '이슬람교도', '무슬림', '이슬람', '기독교인', '기독교', '천주교인',
    '재일교포', '베트남인', '일본인', '탈북민', '남한사람',
    '조선족', '외국인', '내국인', '이민자', '시민', '동남아', '한국인', '한국',
    '노년층', '청년층', '할머니', '할아버지', '젊은여자', '젊은남자', '노인', '청년',
    '동성애자', '트랜스젠더', '성소수자', '레즈비언', '퀴어', '게이', '이성애자',
    '정신장애인', '지적장애인', '장애인', '비장애인',
]

def has_term_as_token(text: str, terms: list) -> tuple[str | None, list]:
    """형태소 분석 후 term이 독립 토큰으로 있는지 확인"""
    result = kiwi.tokenize(text)
    tokens = [t.form for t in result]
    for term in sorted(terms, key=len, reverse=True):
        if term in tokens:
            return term, tokens
    return None, tokens

def has_term_string(text: str, terms: list) -> bool:
    """단순 문자열 포함 여부"""
    return any(t in text for t in sorted(terms, key=len, reverse=True))

print("K-HATERS train 5000개 샘플로 측정 중...")
ds = load_dataset('humane-lab/K-HATERS', split='train')

import random
random.seed(42)
sample = random.sample(list(ds), 5000)

string_match = 0
token_match = 0
cat_counter = Counter()

for row in sample:
    text = row['text'].strip()
    if not text:
        continue
    if has_term_string(text, SWAP_KEYS_ALL):
        string_match += 1
    term, _ = has_term_as_token(text, SWAP_KEYS_ALL)
    if term:
        token_match += 1
        # 카테고리 판별
        if any(term in g for g in [['여성','남성','여자','남자','페미','한남','메갈','한녀','여성들','남성들','여자들','남자들','페미니스트','남성우월주의자']]):
            cat_counter['gender'] += 1
        elif any(term in g for g in [['조선족','외국인','내국인','이민자','시민','동남아','한국인','한국','재일교포','베트남인','일본인','탈북민','남한사람']]):
            cat_counter['ethnicity'] += 1
        elif term in ['노인','청년','노년층','청년층','할머니','할아버지','젊은여자','젊은남자']:
            cat_counter['age'] += 1
        elif term in ['무슬림','이슬람','기독교인','기독교','천주교인','이슬람교도']:
            cat_counter['religion'] += 1
        elif term in ['동성애자','트랜스젠더','성소수자','레즈비언','퀴어','게이','이성애자']:
            cat_counter['sexuality'] += 1
        elif term in ['장애인','비장애인','정신장애인','지적장애인']:
            cat_counter['disability'] += 1

print(f"\n샘플 {len(sample)}개 기준:")
print(f"  문자열 매칭 (현재 방식): {string_match}개 ({100*string_match/len(sample):.1f}%)")
print(f"  형태소 토큰 매칭 (새 방식): {token_match}개 ({100*token_match/len(sample):.1f}%)")
print(f"  유지율: {100*token_match/string_match:.1f}%" if string_match else "")
print()
print("카테고리별 분포 (형태소 기준):")
for cat, cnt in cat_counter.most_common():
    print(f"  {cat}: {cnt}개")
print()
print(f"전체 172K train 추정:")
ratio = token_match / len(sample)
print(f"  형태소 기준 CF 가능 문장: ~{int(172000 * ratio):,}개 ({100*ratio:.1f}%)")

import json, random
from collections import Counter

with open('validity_gated_exp/data/cf_pairs_train.jsonl', encoding='utf-8') as f:
    pairs = [json.loads(l) for l in f]

print(f'총 {len(pairs)}개 CF 쌍\n')

cats = Counter(p['category'] for p in pairs)
print('카테고리별 분포:')
for cat, cnt in cats.most_common():
    print(f'  {cat}: {cnt}개')

print()
print('=== 샘플 15개 ===')
random.seed(42)
for p in random.sample(pairs, 15):
    label_str = 'hate' if p['label'] == 1 else 'normal'
    print(f"[{p['category']}] [{label_str}]  {p['orig_term']} → {p['swap_term']}")
    print(f"  원본: {p['original'][:75]}")
    print(f"  CF  : {p['cf'][:75]}")
    print()

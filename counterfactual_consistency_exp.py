# pip install -q transformers scikit-learn scipy

import os

# ── 여기만 수정 ─────────────────────────────────────────────
BASE_DIR = r'C:\nlp_project'
# ──────────────────────────────────────────────────────────

KOLD_PATH   = os.path.join(BASE_DIR, 'data', 'kold_v1.json')
FLIP_PATH   = os.path.join(BASE_DIR, 'data', 'counterfactual_eval_v3_150_draft.json')
CKPT_DIR    = os.path.join(BASE_DIR, 'checkpoints_v2')
RESULT_PATH = os.path.join(BASE_DIR, 'results_consistency_reg.json')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f'KOLD    : {KOLD_PATH}  exists={os.path.exists(KOLD_PATH)}')
print(f'EvalSet : {FLIP_PATH}  exists={os.path.exists(FLIP_PATH)}')
print(f'Ckpt dir: {CKPT_DIR}')

import urllib.request

# kold_v1.json 없으면 GitHub에서 자동 다운로드 (git 불필요)
if not os.path.exists(KOLD_PATH):
    print('kold_v1.json 없음 → GitHub에서 다운로드 중...')
    url = 'https://raw.githubusercontent.com/boychaboy/KOLD/main/data/kold_v1.json'
    urllib.request.urlretrieve(url, KOLD_PATH)
    print(f'저장 완료 → {KOLD_PATH}')
else:
    print(f'kold_v1.json 이미 존재: {KOLD_PATH}')

MODEL_NAME   = 'klue/roberta-base'
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
LAMBDA       = 0.1       # consistency loss default lambda
SUBSET       = 0         # 0 = full KOLD / int = balanced subsample size
SEEDS        = [42, 123, 456]

# Evaluation-set controls
CORE_ONLY      = False   # False = GPT+Claude 검증 통과 쌍 전체 사용 (122/150)
INCLUDE_SUBTLE = False   # subtle_stereotype 제외 (모호성 높아 main 분석 제외)
VERBOSE_PAIRS  = False

# GPT + Claude 검증에서 양쪽 모두 탈락 판정한 28쌍 (두 LLM 일치)
REJECTED_IDS = {
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34,   # subtle curated 불일치
    95, 96, 97, 99, 100,                        # stereotype_flip draft 불일치
    115, 116, 117, 118, 119, 120,               # stereotype_flip draft 불일치
    136, 137, 139, 143, 145, 146, 148,          # subtle draft 불일치
}


import json, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from scipy import stats
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
print(f'Model  : {MODEL_NAME}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 긴 term 먼저 치환 (부분치환 방지)
IDENTITY_TERMS = sorted([
    # 성별
    '여성', '여자', '남성', '남자', '여성들', '남성들', '여자들', '남자들',
    '페미니스트', '페미', '메갈', '한남', '한녀',
    # 국적/민족
    '조선족', '중국인', '외국인', '이민자', '탈북민', '북한',
    '베트남인', '동남아', '일본인', '재일교포',
    # 종교
    '기독교인', '무슬림', '이슬람', '천주교인',
    # 장애
    '장애인', '정신장애', '지적장애',
    # 성소수자
    '동성애자', '게이', '레즈비언', '트랜스젠더', '성소수자', '퀴어', '성전환자',
    # 연령/기타
    '노인', '노년층', '이주노동자', '난민',
], key=len, reverse=True)

def mask_identity_terms(text, mask_tok):
    masked, found = text, []
    for term in IDENTITY_TERMS:
        if term in masked:
            masked = masked.replace(term, mask_tok)
            found.append(term)
    return masked, found

print(f'Identity terms 등록: {len(IDENTITY_TERMS)}개')

def load_kold(path, subset=0, seed=42):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    examples = [(d['comment'], int(d['OFF'])) for d in raw if d['comment'].strip()]
    if subset:
        rng = random.Random(seed)
        pos = [e for e in examples if e[1] == 1]
        neg = [e for e in examples if e[1] == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        examples = pos[:subset//2] + neg[:subset//2]
        rng.shuffle(examples)
    return examples


def build_eval_pairs(flip_json, core_only=False, include_subtle=False):
    pairs = flip_json['pairs']

    if core_only:
        # 보수적 설정: curated_v3 수동 검수 쌍만 사용
        pairs = [p for p in pairs if p.get('status', 'curated_v3') == 'curated_v3']
    else:
        # GPT + Claude 양쪽 검증 통과 쌍 사용 (탈락 28쌍 제외)
        pairs = [p for p in pairs if p['id'] not in REJECTED_IDS]

    if not include_subtle:
        pairs = [p for p in pairs if p['category'] != 'subtle_stereotype']

    return pairs


set_seed(42)
all_data = load_kold(KOLD_PATH, SUBSET)
random.shuffle(all_data)
cut = int(len(all_data) * 0.8)
train_raw, val_raw = all_data[:cut], all_data[cut:]
print(f'KOLD  total={len(all_data)}  train={len(train_raw)}  val={len(val_raw)}')
print(f'  offensive rate (train): {sum(l for _,l in train_raw)/len(train_raw):.3f}')

with open(FLIP_PATH, encoding='utf-8') as f:
    flip_json = json.load(f)

all_eval_pairs = flip_json['pairs']
eval_pairs = build_eval_pairs(
    flip_json,
    core_only=CORE_ONLY,
    include_subtle=INCLUDE_SUBTLE,
)

print(f'\nEval v{flip_json["meta"]["version"]}: raw={len(all_eval_pairs)} pairs')
print(f'  core_only={CORE_ONLY}  include_subtle={INCLUDE_SUBTLE}')
print(f'  selected pairs = {len(eval_pairs)}')

from collections import Counter
selected_counts = Counter(p['category'] for p in eval_pairs)
for cat in ('adversarial_clean', 'stereotype_flip_strict', 'subtle_stereotype'):
    if selected_counts.get(cat):
        print(f'  {cat}: {selected_counts[cat]}쌍')

if CORE_ONLY:
    curated = sum(1 for p in eval_pairs if p.get('status') == 'curated_v3')
    print(f'  curated_v3 pairs used: {curated}')
else:
    validated = len(eval_pairs)
    rejected_applied = sum(1 for p in all_eval_pairs if p['id'] in REJECTED_IDS)
    print(f'  LLM-validated pairs used: {validated}  (rejected {rejected_applied}쌍 제외)')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class KOLDDataset(Dataset):
    """
    masking=True  ->  각 샘플에 masked 버전 추가 반환
                      (consistency loss & masking aug 용)
    """
    def __init__(self, examples, tokenizer, max_len, masking=False):
        self.tok, self.max_len, self.masking = tokenizer, max_len, masking
        self.items = []
        for text, label in examples:
            masked, found = mask_identity_terms(text, tokenizer.mask_token)
            self.items.append({
                'text': text, 'masked': masked, 'label': label,
                'has_id': len(found) > 0,
            })

    def _enc(self, text):
        e = self.tok(text, max_length=self.max_len,
                     padding='max_length', truncation=True, return_tensors='pt')
        return e['input_ids'].squeeze(0), e['attention_mask'].squeeze(0)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        ids, mask = self._enc(it['text'])
        out = {
            'input_ids':      ids,
            'attention_mask': mask,
            'label':          torch.tensor(it['label'], dtype=torch.long),
            'has_id':         torch.tensor(it['has_id'],  dtype=torch.bool),
        }
        if self.masking:
            m_ids, m_mask = self._enc(it['masked'])
            out['mask_input_ids']      = m_ids
            out['mask_attention_mask'] = m_mask
        return out

print('Dataset 클래스 정의 완료')

class HateDetector(nn.Module):
    """KLUE-RoBERTa [CLS] -> Dropout -> Linear(2)"""
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))   # (B, 2)

    def probs(self, input_ids, attention_mask):
        return F.softmax(self.forward(input_ids, attention_mask), dim=-1)

print('모델 클래스 정의 완료')

def cons_loss_kl(p_orig, p_mask):
    """대칭 KL: (KL(orig||mask) + KL(mask||orig)) / 2"""
    kl_fwd = F.kl_div(p_mask.log(), p_orig, reduction='batchmean')
    kl_bwd = F.kl_div(p_orig.log(), p_mask, reduction='batchmean')
    return (kl_fwd + kl_bwd) / 2


def train_epoch(model, loader, optimizer, use_masking, use_cons_reg, lam):
    model.train()
    s_loss = s_hate = s_cons = 0.0
    for batch in tqdm(loader, desc='  train', leave=False):
        ids    = batch['input_ids'].to(device)
        mask   = batch['attention_mask'].to(device)
        y      = batch['label'].to(device)
        has_id = batch['has_id'].to(device)

        optimizer.zero_grad()
        logits    = model(ids, mask)
        hate_loss = F.cross_entropy(logits, y)
        loss      = hate_loss
        c_val     = torch.tensor(0.0, device=device)

        if (use_masking or use_cons_reg) and 'mask_input_ids' in batch:
            m_ids  = batch['mask_input_ids'].to(device)
            m_mask = batch['mask_attention_mask'].to(device)

            if use_masking:                           # masking augmentation
                loss = loss + F.cross_entropy(model(m_ids, m_mask), y)

            if use_cons_reg and has_id.any():         # consistency regularization
                p_o   = model.probs(ids[has_id],   mask[has_id])
                p_m   = model.probs(m_ids[has_id], m_mask[has_id])
                c_val = cons_loss_kl(p_o, p_m)
                loss  = loss + lam * c_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        s_loss += loss.item(); s_hate += hate_loss.item(); s_cons += c_val.item()

    n = len(loader)
    return s_loss/n, s_hate/n, s_cons/n


def eval_kold(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval', leave=False):
            logits = model(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(batch['label'].tolist())
    return f1_score(labels, preds, average='macro')


print('학습/평가 함수 정의 완료')

def predict_text(model, text):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                        truncation=True, return_tensors='pt')
        logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        prob   = F.softmax(logits, dim=-1)[0, 1].item()
    return int(prob >= 0.5), round(prob, 3)


def eval_stereotype_flip(model, tag=''):
    raw = []
    for p in eval_pairs:
        op, oprob = predict_text(model, p['original'])
        cp, cprob = predict_text(model, p['counterfactual'])
        raw.append({
            'id': p['id'], 'category': p['category'],
            'expected': p['expected_label'],
            'orig_pred': op, 'cf_pred': cp,
            'orig_prob': oprob, 'cf_prob': cprob,
            'consistent': op == cp,
            'orig_correct': op == p['expected_label'],
            'cf_correct': cp == p['expected_label'],
            'pair_correct': (op == p['expected_label']) and (cp == p['expected_label']),
            'identity_orig': p.get('identity_orig', ''),
            'trap': p.get('trap', ''),
            'status': p.get('status', 'unknown'),
        })

    total = len(raw)
    ccs   = sum(r['consistent'] for r in raw) / total if total else 0.0

    cat_stats = {}
    for cat in ('adversarial_clean', 'stereotype_flip_strict', 'subtle_stereotype'):
        sub = [r for r in raw if r['category'] == cat]
        if not sub:
            continue
        n = len(sub)
        c = sum(r['consistent'] for r in sub) / n
        if cat == 'adversarial_clean':
            cat_stats[cat] = {
                'n': n,
                'ccs': c,
                'fpr_orig': sum(r['orig_pred'] == 1 for r in sub) / n,
                'fpr_cf':   sum(r['cf_pred'] == 1 for r in sub) / n,
                'pair_acc': sum(r['pair_correct'] for r in sub) / n,
            }
        else:
            cat_stats[cat] = {
                'n': n,
                'ccs': c,
                'tpr_orig': sum(r['orig_correct'] for r in sub) / n,
                'tpr_cf':   sum(r['cf_correct'] for r in sub) / n,
                'pair_acc': sum(r['pair_correct'] for r in sub) / n,
            }

    print(f'  [{tag}]  CCS={ccs:.4f}  FlipRate={1-ccs:.4f}')
    for cat, s in cat_stats.items():
        if cat == 'adversarial_clean':
            print(
                f"  [{cat}  n={s['n']}]  CCS={s['ccs']:.4f}  "
                f"FPR(orig)={s['fpr_orig']:.4f}  FPR(cf)={s['fpr_cf']:.4f}  PairAcc={s['pair_acc']:.4f}"
            )
        else:
            print(
                f"  [{cat}  n={s['n']}]  CCS={s['ccs']:.4f}  "
                f"TPR(orig)={s['tpr_orig']:.4f}  TPR(cf)={s['tpr_cf']:.4f}  PairAcc={s['pair_acc']:.4f}"
            )

    if VERBOSE_PAIRS:
        bad = [r for r in raw if not r['pair_correct']][:10]
        if bad:
            print('  sample difficult pairs:')
            for r in bad:
                print(f"    id={r['id']} cat={r['category']} orig={r['orig_pred']} cf={r['cf_pred']} trap={r['trap']}")

    return {'ccs': ccs, 'flip_rate': 1-ccs, 'category_stats': cat_stats, 'raw': raw}


print('평가 함수 정의 완료')



def run_experiment(tag, use_masking, use_cons_reg, lam=LAMBDA,
                   seeds=None, n_epochs=EPOCHS):
    if seeds is None:
        seeds = SEEDS
    needs_mask = use_masking or use_cons_reg
    metrics = {
        'kold_f1': [], 'ccs': [], 'flip_rate': [], 'fpr_orig': [],
        'strict_tpr_orig': [], 'strict_tpr_cf': [], 'strict_pair_acc': [],
        'clean_pair_acc': [],
    }

    for seed in seeds:
        print(f'\n[{tag}] seed={seed}  lam={lam}')
        set_seed(seed)

        tr_ds = KOLDDataset(train_raw, tokenizer, MAX_LEN, masking=needs_mask)
        va_ds = KOLDDataset(val_raw, tokenizer, MAX_LEN, masking=False)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=torch.cuda.is_available())
        va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=torch.cuda.is_available())

        model = HateDetector(MODEL_NAME).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)

        best_f1, best_state = 0.0, None
        for ep in range(1, n_epochs + 1):
            tl, hl, cl = train_epoch(model, tr_dl, opt,
                                     use_masking, use_cons_reg, lam)
            vf1 = eval_kold(model, va_dl)
            print(f'  ep{ep}: total={tl:.4f} hate={hl:.4f} cons={cl:.4f} | val_F1={vf1:.4f}')
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                ckpt_path = os.path.join(CKPT_DIR, f"{tag.replace(' ', '_')}_seed{seed}.pt")
                torch.save(best_state, ckpt_path)

        model.load_state_dict(best_state)
        vf1_final = eval_kold(model, va_dl)
        sf = eval_stereotype_flip(model, tag=f'{tag} seed={seed}')
        print(f'  best val Macro-F1: {vf1_final:.4f}')

        metrics['kold_f1'].append(vf1_final)
        metrics['ccs'].append(sf['ccs'])
        metrics['flip_rate'].append(sf['flip_rate'])

        clean = sf['category_stats'].get('adversarial_clean', {})
        strict = sf['category_stats'].get('stereotype_flip_strict', {})

        if 'fpr_orig' in clean:
            metrics['fpr_orig'].append(clean['fpr_orig'])
        if 'pair_acc' in clean:
            metrics['clean_pair_acc'].append(clean['pair_acc'])
        if 'tpr_orig' in strict:
            metrics['strict_tpr_orig'].append(strict['tpr_orig'])
        if 'tpr_cf' in strict:
            metrics['strict_tpr_cf'].append(strict['tpr_cf'])
        if 'pair_acc' in strict:
            metrics['strict_pair_acc'].append(strict['pair_acc'])

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def _s(lst):
        return f'{np.mean(lst):.4f}+/-{np.std(lst):.4f}' if lst else 'N/A'

    print(f'\n{"="*60}')
    print(f'  [{tag}]  {len(seeds)}-seed 요약')
    print(f'  KOLD Macro-F1     : {_s(metrics["kold_f1"])}')
    print(f'  CCS               : {_s(metrics["ccs"])}')
    print(f'  Flip Rate         : {_s(metrics["flip_rate"])}')
    print(f'  Clean FPR(orig)   : {_s(metrics["fpr_orig"])}')
    print(f'  Clean PairAcc     : {_s(metrics["clean_pair_acc"])}')
    print(f'  Strict TPR(orig)  : {_s(metrics["strict_tpr_orig"])}')
    print(f'  Strict TPR(cf)    : {_s(metrics["strict_tpr_cf"])}')
    print(f'  Strict PairAcc    : {_s(metrics["strict_pair_acc"])}')
    print(f'{"="*60}')

    return metrics


ABLATIONS = [
    dict(tag='Baseline',          use_masking=False, use_cons_reg=False, lam=0.0),
    dict(tag='Masking Aug Only',  use_masking=True,  use_cons_reg=False, lam=0.0),
    dict(tag='Cons Reg Only',     use_masking=False, use_cons_reg=True,  lam=LAMBDA),
    dict(tag='Full Model',        use_masking=True,  use_cons_reg=True,  lam=LAMBDA),
]

all_results = {}
for exp in ABLATIONS:
    print(f"\n{'#'*60}\n  실험: {exp['tag']}\n{'#'*60}")
    all_results[exp['tag']] = run_experiment(**exp)

for lam in [0.05, 0.1, 0.2]:
    key = f'Full_lam={lam}'
    all_results[key] = run_experiment(
        tag=key, use_masking=True, use_cons_reg=True,
        lam=lam, n_epochs=3,
    )

def _fmt(lst):
    return f'{np.mean(lst):.4f}+/-{np.std(lst):.4f}' if lst else 'N/A'

print('\n' + '='*126)
print(f"  {'Model':<24} {'Macro-F1':>16} {'StrictTPR':>16} {'CleanFPR':>16} {'CCS':>16} {'StrictPairAcc':>18}")
print('='*126)
for name, r in all_results.items():
    print(
        f"  {name:<24}  {_fmt(r['kold_f1']):>16}  {_fmt(r['strict_tpr_orig']):>16}  "
        f"{_fmt(r['fpr_orig']):>16}  {_fmt(r['ccs']):>16}  {_fmt(r['strict_pair_acc']):>18}"
    )

if 'Baseline' in all_results and 'Full Model' in all_results:
    base = all_results['Baseline']['strict_tpr_orig']
    full = all_results['Full Model']['strict_tpr_orig']
    if base and full and len(base) == len(full):
        t, p = stats.ttest_rel(full, base)
        print(
            f"\n  [Pilot check] Baseline vs Full Model Strict TPR  t={t:.4f}  p={p:.4f}  "
            f"{'*유의*' if p < 0.05 else '비유의'} (alpha=0.05)"
        )

save_data = {}
for k, d in all_results.items():
    save_data[k] = d
with open(RESULT_PATH, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)
print(f'\n  결과 저장 -> {RESULT_PATH}')



ckpt = os.path.join(CKPT_DIR, 'Full_Model_seed42.pt')
if not os.path.exists(ckpt):
    print(f'checkpoint 없음: {ckpt}\n=> Cell 11을 먼저 실행하세요')
else:
    model_ea = HateDetector(MODEL_NAME).to(device)
    model_ea.load_state_dict(torch.load(ckpt, map_location=device))
    sf_ea = eval_stereotype_flip(model_ea, tag='Error Analysis')
    raw   = sf_ea['raw']

    flips = [r for r in raw if not r['consistent']]
    print(f'\n[Flip 발생 사례]  {len(flips)}/{len(raw)}쌍 — 처음 10건')
    for r in flips[:10]:
        p = eval_pairs[r['id'] - 1]
        print(f"\n  id={r['id']:>3}  cat={r['category']}")
        print(f"  orig : {p['original'][:70]}")
        print(f"  cf   : {p['counterfactual'][:70]}")
        print(f"  pred : orig={r['orig_pred']}({r['orig_prob']}) -> "
              f"cf={r['cf_pred']}({r['cf_prob']})  expected={r['expected']}")
        print(f"  trap : {r['trap']}")

    adv_fp = [r for r in raw
              if r['category'] == 'adversarial_clean' and r['orig_pred'] == 1]
    print(f'\n[Adversarial Clean FP]  {len(adv_fp)}건 — 처음 5건')
    for r in adv_fp[:5]:
        p = eval_pairs[r['id'] - 1]
        print(f"  id={r['id']:>3}  {p['original'][:75]}")
        print(f"         trap: {r['trap']}")

    del model_ea; gc.collect(); torch.cuda.empty_cache()

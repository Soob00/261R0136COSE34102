# ============================================================
# 한국어 혐오 탐지 - Counterfactual Consistency Regularization
# proposal_revised.md 기반 구현
# Google Colab 실행용
# ============================================================
#
# [Colab 준비]
# 1. 이 파일을 셀별로 Colab에 붙여넣거나, .ipynb로 변환해서 사용
# 2. counterfactual_eval_v3_150_draft.json → Colab에 업로드
# 3. KOLD 데이터 다운로드: https://github.com/boychaboy/KOLD
#    (kold_train.tsv, kold_test.tsv 업로드)
# 4. GPU 런타임 사용 권장 (런타임 → 런타임 유형 변경 → T4 GPU)
# ============================================================


# %%
# ======================== Cell 1: 패키지 설치 ========================
# !pip install transformers datasets torch scikit-learn scipy tqdm


# %%
# ======================== Cell 2: 임포트 & 설정 ========================
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── 실험 설정 ──────────────────────────────────────────────
CONFIG = {
    "model_name": "klue/roberta-base",   # klue/roberta-large로 교체 가능
    "max_length": 128,
    "batch_size": 16,
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "lambda_cons": 0.1,                  # 기본 λ
    "seeds": [42, 123, 456],
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ── 파일 경로 (Colab 업로드 후 경로로 수정) ──
    "eval_path": "counterfactual_eval_v3_150_draft.json",
    "kold_train_path": "kold_train.tsv",   # 또는 .jsonl
    "kold_test_path":  "kold_test.tsv",
}

print(f"Device : {CONFIG['device']}")
print(f"Model  : {CONFIG['model_name']}")


# %%
# ======================== Cell 3: Identity Term 목록 ========================
# KOLD 훈련 데이터에서 마스킹할 한국어 identity term 목록
IDENTITY_TERMS = sorted([
    # 성별
    "여성", "여자", "남성", "남자", "여성들", "남성들", "여자들", "남자들",
    "페미니스트", "페미", "메갈", "한남", "한녀",
    # 국적/민족
    "조선족", "중국인", "외국인", "이민자", "탈북민", "북한",
    "베트남인", "동남아", "일본인", "재일교포",
    # 종교
    "기독교인", "무슬림", "이슬람", "천주교인", "불교신자",
    # 장애
    "장애인", "정신장애", "지적장애", "시각장애", "청각장애",
    # 성소수자
    "동성애자", "게이", "레즈비언", "트랜스젠더", "성소수자", "퀴어", "성전환자",
    # 연령
    "노인", "노년층", "청년", "20대", "30대",
    # 기타
    "이주노동자", "난민",
], key=len, reverse=True)  # 긴 term 먼저 치환 (부분치환 방지)


def mask_identity_terms(text, mask_token="[MASK]"):
    """텍스트에서 identity term을 mask_token으로 치환. (found 목록도 반환)"""
    masked = text
    found = []
    for term in IDENTITY_TERMS:
        if term in masked:
            masked = masked.replace(term, mask_token)
            found.append(term)
    return masked, found


# %%
# ======================== Cell 4: KOLD 데이터 로딩 ========================
def load_kold_tsv(path):
    """
    KOLD GitHub TSV 형식 로딩.
    컬럼: id, comment, off, tgt, form, grp  (헤더 있음)
    off 값: 'YES'/'NO'  또는  1/0
    """
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            text = row.get('comment', row.get('text', '')).strip()
            off_val = row.get('off', row.get('label', 'NO')).strip()
            # YES/NO / OFF/NOT / 1/0 모두 처리
            label = 1 if off_val.upper() in ('YES', 'OFF', '1', 'TRUE') else 0
            if text:
                samples.append({'text': text, 'label': label})
    return samples


def load_kold_jsonl(path):
    """KOLD JSONL 형식 로딩 (대안)."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get('comment', item.get('text', '')).strip()
            off_val = item.get('off', item.get('label', 0))
            label = 1 if str(off_val).upper() in ('YES', 'OFF', '1', 'TRUE') else 0
            if text:
                samples.append({'text': text, 'label': label})
    return samples


def load_kold(path):
    """확장자에 따라 자동으로 로더 선택."""
    if path.endswith('.jsonl') or path.endswith('.json'):
        return load_kold_jsonl(path)
    else:
        return load_kold_tsv(path)


# %%
# ======================== Cell 5: Dataset 클래스 ========================
class KOLDDataset(Dataset):
    """
    KOLD 훈련/검증/테스트 데이터셋.
    masking=True면 각 샘플에 masked 버전도 포함 (consistency loss용).
    """
    def __init__(self, samples, tokenizer, max_length=128, masking=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masking = masking

        self.items = []
        for s in samples:
            text = s['text']
            label = s['label']
            masked_text, found = mask_identity_terms(text, tokenizer.mask_token)
            self.items.append({
                'text': text,
                'masked_text': masked_text,
                'label': label,
                'has_identity': len(found) > 0,
            })

    def _encode(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        enc = self._encode(it['text'])
        out = {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(it['label'], dtype=torch.long),
            'has_identity':   torch.tensor(it['has_identity'], dtype=torch.bool),
        }
        if self.masking:
            m_enc = self._encode(it['masked_text'])
            out['mask_input_ids']      = m_enc['input_ids'].squeeze(0)
            out['mask_attention_mask'] = m_enc['attention_mask'].squeeze(0)
        return out


# %%
# ======================== Cell 6: 모델 ========================
class HateDetector(nn.Module):
    """KLUE-RoBERTa [CLS] → Linear 분류기"""
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

    def get_probs(self, input_ids, attention_mask):
        return F.softmax(self.forward(input_ids, attention_mask), dim=-1)


# %%
# ======================== Cell 7: 손실 함수 ========================
def consistency_loss(orig_probs, mask_probs):
    """
    대칭 KL divergence: (KL(orig||mask) + KL(mask||orig)) / 2
    원본과 마스킹 문장의 예측 분포를 일치시킴.
    """
    kl_fwd = F.kl_div(mask_probs.log(), orig_probs, reduction='batchmean')
    kl_bwd = F.kl_div(orig_probs.log(), mask_probs, reduction='batchmean')
    return (kl_fwd + kl_bwd) / 2


# %%
# ======================== Cell 8: 학습 함수 ========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device,
                    use_masking=False, use_cons_reg=False, lambda_cons=0.1):
    """
    use_masking  : masked 문장에도 L_hate 적용 (데이터 증강)
    use_cons_reg : identity term 포함 샘플에 L_cons 적용
    """
    model.train()
    sum_loss = sum_hate = sum_cons = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)
        has_id         = batch['has_identity'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        hate_loss = F.cross_entropy(logits, labels)
        loss = hate_loss
        cons_val = torch.tensor(0.0, device=device)

        if (use_masking or use_cons_reg) and 'mask_input_ids' in batch:
            m_ids  = batch['mask_input_ids'].to(device)
            m_mask = batch['mask_attention_mask'].to(device)

            if use_masking:
                m_logits = model(m_ids, m_mask)
                loss = loss + F.cross_entropy(m_logits, labels)

            if use_cons_reg and has_id.any():
                orig_p = model.get_probs(input_ids[has_id], attention_mask[has_id])
                mask_p = model.get_probs(m_ids[has_id], m_mask[has_id])
                cons_val = consistency_loss(orig_p, mask_p)
                loss = loss + lambda_cons * cons_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sum_loss += loss.item()
        sum_hate += hate_loss.item()
        sum_cons += cons_val.item()

    n = len(loader)
    return sum_loss / n, sum_hate / n, sum_cons / n


def eval_kold(model, loader, device):
    """KOLD 테스트셋 → Macro-F1"""
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval kold", leave=False):
            logits = model(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device))
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            labels_all.extend(batch['label'].tolist())
    return f1_score(labels_all, preds_all, average='macro')


# %%
# ======================== Cell 9: Stereotype Flip v3 평가 ========================
def eval_stereotype_flip(model, pairs, tokenizer, device, max_length=128):
    """
    Returns dict:
      ccs         : Counterfactual Consistency Score (전체)
      flip_rate   : 1 - ccs
      category_stats : 카테고리별 상세 지표
      raw         : 샘플별 결과 리스트
    """
    model.eval()
    raw = []

    with torch.no_grad():
        for p in tqdm(pairs, desc="  eval SF", leave=False):
            def enc(text):
                e = tokenizer(text, max_length=max_length,
                              padding='max_length', truncation=True,
                              return_tensors='pt')
                return e['input_ids'].to(device), e['attention_mask'].to(device)

            orig_ids, orig_mask = enc(p['original'])
            cf_ids,   cf_mask   = enc(p['counterfactual'])

            orig_pred = model(orig_ids, orig_mask).argmax(-1).item()
            cf_pred   = model(cf_ids,   cf_mask).argmax(-1).item()
            expected  = p['expected_label']

            raw.append({
                'id':          p['id'],
                'category':    p['category'],
                'expected':    expected,
                'orig_pred':   orig_pred,
                'cf_pred':     cf_pred,
                'consistent':  orig_pred == cf_pred,
                'orig_correct': orig_pred == expected,
                'cf_correct':   cf_pred  == expected,
            })

    total = len(raw)
    ccs = sum(r['consistent'] for r in raw) / total

    cat_stats = {}
    for cat in ('adversarial_clean', 'stereotype_flip_strict', 'subtle_stereotype'):
        sub = [r for r in raw if r['category'] == cat]
        if not sub:
            continue
        n = len(sub)
        cat_ccs = sum(r['consistent'] for r in sub) / n

        if cat == 'adversarial_clean':
            # expected=0  →  FPR = pred==1 비율
            cat_stats[cat] = {
                'n':       n,
                'ccs':     cat_ccs,
                'fpr_orig': sum(r['orig_pred'] == 1 for r in sub) / n,
                'fpr_cf':   sum(r['cf_pred']   == 1 for r in sub) / n,
            }
        else:
            # expected=1  →  TPR = pred==1 비율
            cat_stats[cat] = {
                'n':        n,
                'ccs':      cat_ccs,
                'tpr_orig': sum(r['orig_correct'] for r in sub) / n,
                'tpr_cf':   sum(r['cf_correct']   for r in sub) / n,
            }

    return {'ccs': ccs, 'flip_rate': 1 - ccs, 'category_stats': cat_stats, 'raw': raw}


def print_sf(res, tag=""):
    cs = res['category_stats']
    print(f"\n{'─'*55}")
    print(f"  Stereotype Flip v3  {tag}")
    print(f"  CCS(전체)={res['ccs']:.4f}  FlipRate={res['flip_rate']:.4f}")
    if 'adversarial_clean' in cs:
        c = cs['adversarial_clean']
        print(f"  [Adversarial Clean n={c['n']}]"
              f"  CCS={c['ccs']:.4f}  FPR(orig)={c['fpr_orig']:.4f}  FPR(cf)={c['fpr_cf']:.4f}")
    if 'stereotype_flip_strict' in cs:
        c = cs['stereotype_flip_strict']
        print(f"  [Stereotype Strict  n={c['n']}]"
              f"  CCS={c['ccs']:.4f}  TPR(orig)={c['tpr_orig']:.4f}  TPR(cf)={c['tpr_cf']:.4f}")
    if 'subtle_stereotype' in cs:
        c = cs['subtle_stereotype']
        print(f"  [Subtle Stereotype  n={c['n']}]"
              f"  CCS={c['ccs']:.4f}  (탐색적 분석)")


# %%
# ======================== Cell 10: 단일 실험 실행 ========================
def run_experiment(tag, train_raw, val_raw, test_raw, eval_pairs,
                   tokenizer, device, config,
                   use_masking, use_cons_reg, lambda_cons,
                   seeds=None):
    """지정된 설정으로 N-seed 실험 수행."""
    if seeds is None:
        seeds = config['seeds']

    needs_mask = use_masking or use_cons_reg
    metrics = {'kold_f1': [], 'ccs': [], 'flip_rate': [], 'fpr_orig': []}

    for seed in seeds:
        print(f"\n[{tag}] seed={seed}")
        set_seed(seed)

        tr_ds = KOLDDataset(train_raw, tokenizer, config['max_length'], masking=needs_mask)
        va_ds = KOLDDataset(val_raw,   tokenizer, config['max_length'], masking=False)
        te_ds = KOLDDataset(test_raw,  tokenizer, config['max_length'], masking=False)

        tr_dl = DataLoader(tr_ds, batch_size=config['batch_size'], shuffle=True,  num_workers=2, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        te_dl = DataLoader(te_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

        model = HateDetector(config['model_name']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])

        best_val_f1, best_state = 0.0, None

        for epoch in range(config['num_epochs']):
            loss, h_l, c_l = train_one_epoch(
                model, tr_dl, optimizer, device,
                use_masking=use_masking,
                use_cons_reg=use_cons_reg,
                lambda_cons=lambda_cons,
            )
            val_f1 = eval_kold(model, va_dl, device)
            print(f"  ep{epoch+1}: total={loss:.4f} hate={h_l:.4f} cons={c_l:.4f} | val_F1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_f1 = eval_kold(model, te_dl, device)
        sf = eval_stereotype_flip(model, eval_pairs, tokenizer, device, config['max_length'])

        print(f"  KOLD test Macro-F1: {test_f1:.4f}")
        print_sf(sf, tag=f"[{tag} seed={seed}]")

        metrics['kold_f1'].append(test_f1)
        metrics['ccs'].append(sf['ccs'])
        metrics['flip_rate'].append(sf['flip_rate'])
        fpr = sf['category_stats'].get('adversarial_clean', {}).get('fpr_orig')
        if fpr is not None:
            metrics['fpr_orig'].append(fpr)

    # 요약
    def _fmt(lst):
        return f"{np.mean(lst):.4f} ± {np.std(lst):.4f}" if lst else "N/A"

    print(f"\n{'='*55}")
    print(f"  [{tag}] {len(seeds)}-seed 요약")
    print(f"  KOLD Macro-F1 : {_fmt(metrics['kold_f1'])}")
    print(f"  CCS           : {_fmt(metrics['ccs'])}")
    print(f"  Flip Rate     : {_fmt(metrics['flip_rate'])}")
    print(f"  FPR (adv)     : {_fmt(metrics['fpr_orig'])}")

    return metrics


# %%
# ======================== Cell 11: 데이터 로딩 ========================
# ── Evaluation 데이터 ──
with open(CONFIG['eval_path'], 'r', encoding='utf-8') as f:
    eval_json = json.load(f)
eval_pairs = eval_json['pairs']
print(f"Eval pairs: {len(eval_pairs)}")
for cat, cnt in eval_json['meta']['categories'].items():
    print(f"  {cat}: {cnt['count']}")

# ── KOLD 훈련/테스트 데이터 ──
# 방법 A: GitHub에서 직접 다운로드 후 업로드 (tsv/jsonl)
#   https://github.com/boychaboy/KOLD  →  data/ 폴더

# 방법 B: Colab에서 직접 wget (public URL이 있다면)
# !wget -q URL -O kold_train.tsv
# !wget -q URL -O kold_test.tsv

train_all = load_kold(CONFIG['kold_train_path'])
test_raw  = load_kold(CONFIG['kold_test_path'])

# train → 90/10 split (val 용)
random.seed(42)
random.shuffle(train_all)
split = int(0.9 * len(train_all))
train_raw = train_all[:split]
val_raw   = train_all[split:]

print(f"KOLD  train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")
print(f"  offensive rate (train): {sum(s['label'] for s in train_raw)/len(train_raw):.3f}")

# ── Tokenizer ──
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
device = torch.device(CONFIG['device'])


# %%
# ======================== Cell 12: 4가지 Ablation 실험 ========================
# proposal Section 5.1 Baselines:
#   (1) Baseline               : L_hate only
#   (2) Masking Aug Only       : L_hate(orig) + L_hate(masked)
#   (3) Cons Reg Only          : L_hate(orig) + λ·L_cons
#   (4) Full Model             : L_hate(orig) + L_hate(masked) + λ·L_cons

ABLATIONS = [
    dict(tag="Baseline",          use_masking=False, use_cons_reg=False, lambda_cons=0.0),
    dict(tag="Masking Aug Only",  use_masking=True,  use_cons_reg=False, lambda_cons=0.0),
    dict(tag="Cons Reg Only",     use_masking=False, use_cons_reg=True,  lambda_cons=CONFIG['lambda_cons']),
    dict(tag="Full Model",        use_masking=True,  use_cons_reg=True,  lambda_cons=CONFIG['lambda_cons']),
]

all_results = {}
for exp in ABLATIONS:
    print(f"\n{'#'*60}")
    print(f"  실험: {exp['tag']}  (λ={exp['lambda_cons']})")
    print(f"{'#'*60}")
    all_results[exp['tag']] = run_experiment(
        **exp,
        train_raw=train_raw, val_raw=val_raw, test_raw=test_raw,
        eval_pairs=eval_pairs, tokenizer=tokenizer,
        device=device, config=CONFIG,
    )


# %%
# ======================== Cell 13: λ Sensitivity 실험 ========================
# proposal Section 5.2: λ ∈ {0.05, 0.1, 0.2}

print("\n\n" + "="*60)
print("  λ Sensitivity (Full Model)")
print("="*60)

lambda_cfg = {**CONFIG, 'num_epochs': 3}   # sensitivity는 3 epoch으로 경량화

for lam in [0.05, 0.1, 0.2]:
    key = f"Full_λ={lam}"
    all_results[key] = run_experiment(
        tag=key,
        use_masking=True, use_cons_reg=True, lambda_cons=lam,
        train_raw=train_raw, val_raw=val_raw, test_raw=test_raw,
        eval_pairs=eval_pairs, tokenizer=tokenizer,
        device=device, config=lambda_cfg,
    )


# %%
# ======================== Cell 14: 최종 결과 테이블 ========================
def _m(lst):
    return (np.mean(lst), np.std(lst)) if lst else (float('nan'), float('nan'))

print("\n\n" + "="*80)
print(f"  {'Model':<28} {'Macro-F1':>14} {'CCS':>14} {'FlipRate':>14} {'FPR(adv)':>12}")
print("="*80)
for name, res in all_results.items():
    f1_m, f1_s   = _m(res['kold_f1'])
    ccs_m, ccs_s = _m(res['ccs'])
    fl_m, fl_s   = _m(res['flip_rate'])
    fp_m, fp_s   = _m(res['fpr_orig']) if res['fpr_orig'] else (float('nan'), float('nan'))
    print(f"  {name:<28}"
          f"  {f1_m:.4f}±{f1_s:.4f}"
          f"  {ccs_m:.4f}±{ccs_s:.4f}"
          f"  {fl_m:.4f}±{fl_s:.4f}"
          f"  {fp_m:.4f}±{fp_s:.4f}")

# ── 통계 검정: Baseline vs Full Model (paired t-test on CCS) ──
if 'Baseline' in all_results and 'Full Model' in all_results:
    print("\n  [H2 검증] Baseline vs Full Model CCS paired t-test")
    t, p = stats.ttest_rel(
        all_results['Full Model']['ccs'],
        all_results['Baseline']['ccs'],
    )
    print(f"  t={t:.4f}  p={p:.4f}  {'*유의*' if p < 0.05 else '비유의'} (α=0.05)")


# %%
# ======================== Cell 15: Error Analysis ========================
# Full Model의 마지막 seed 결과로 오류 분석
# (run_experiment가 raw results를 저장하지 않으므로, 단독으로 재평가)

print("\n\n" + "="*60)
print("  Error Analysis  (Full Model, seed=42)")
print("="*60)

set_seed(42)
needs_mask = True

tr_ds = KOLDDataset(train_raw, tokenizer, CONFIG['max_length'], masking=True)
va_ds = KOLDDataset(val_raw,   tokenizer, CONFIG['max_length'], masking=False)
tr_dl = DataLoader(tr_ds, batch_size=CONFIG['batch_size'], shuffle=True,  num_workers=2, pin_memory=True)
va_dl = DataLoader(va_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

model_ea = HateDetector(CONFIG['model_name']).to(device)
opt_ea   = torch.optim.AdamW(model_ea.parameters(),
                              lr=CONFIG['learning_rate'],
                              weight_decay=CONFIG['weight_decay'])

best_val, best_state = 0.0, None
for epoch in range(CONFIG['num_epochs']):
    train_one_epoch(model_ea, tr_dl, opt_ea, device,
                    use_masking=True, use_cons_reg=True,
                    lambda_cons=CONFIG['lambda_cons'])
    vf1 = eval_kold(model_ea, va_dl, device)
    if vf1 > best_val:
        best_val = vf1
        best_state = {k: v.clone() for k, v in model_ea.state_dict().items()}

model_ea.load_state_dict(best_state)
sf_ea = eval_stereotype_flip(model_ea, eval_pairs, tokenizer, device, CONFIG['max_length'])

# ── 오류 사례 출력 ──
print("\n[Flip 발생 사례 (orig_pred ≠ cf_pred)] — 최대 10건")
flips = [r for r in sf_ea['raw'] if r['flipped']]
for r in flips[:10]:
    p = eval_pairs[r['id'] - 1]
    print(f"\n  id={r['id']}  cat={r['category']}")
    print(f"  orig: {p['original'][:60]}")
    print(f"  cf  : {p['counterfactual'][:60]}")
    print(f"  pred: orig={r['orig_pred']} → cf={r['cf_pred']}  (expected={r['expected']})")
    print(f"  trap: {p.get('trap', '')}")

print(f"\n[Adversarial Clean FP 사례] — 최대 5건")
adv_fp = [r for r in sf_ea['raw']
          if r['category'] == 'adversarial_clean' and r['orig_pred'] == 1]
for r in adv_fp[:5]:
    p = eval_pairs[r['id'] - 1]
    print(f"\n  id={r['id']}  orig: {p['original'][:70]}")
    print(f"  trap: {p.get('trap', '')}")

import re
from collections import Counter
from typing import List

_COT_RE     = re.compile(r'\[cot_start\](.*?)\[cot_end\]', re.DOTALL)
_SECTION_RE = re.compile(r'\*\*([^\n*]+)\*\*\n([^\n*][^*]*?)(?=\n\n\*\*|\Z)', re.DOTALL)

def parse_response(text: str) -> dict:
    '''Split a generated string into (cot, answer). All fields are best-effort.'''
    m = _COT_RE.search(text)
    if not m:
        return {'has_cot': False, 'cot': '', 'answer': text.strip()}
    cot = m.group(1)
    answer = text[m.end():].strip()
    return {'has_cot': True, 'cot': cot, 'answer': answer}

def format_score(cot: str) -> float:
    if not cot:
        return -1.0
    score = 0.0
    if cot.endswith('\n'):
        score -= 0.4
    
    sections = _SECTION_RE.findall(cot)
    n = len(sections)
    if   n == 0: score -= 0.6
    elif n == 1: score += 0.2
    elif n == 2: score += 0.6
    elif n <= 5: score += 0.8
    else:        score += 0.4
    if sections:
        good = sum(1 for _, body in sections if len(body.strip()) >= 20)
        score += 0.3 * (good / len(sections))
    if '**' not in cot:
        score -= 0.3
    return score

def length_score(cot: str, agreement: float, target_short=200, target_long=1200) -> float:
    L = len(cot)
    target = target_short + (target_long - target_short) * (1.0 - agreement)
    
    rel = (L - target) / max(target, 1.0)
    return 0.5 * (1.0 - min(rel * rel, 4.0) / 4.0) - 0.1

def _normalize_answer(a: str) -> str:
    return re.sub(r'\s+', ' ', a.strip().lower())[:200]

def compute_group_rewards(responses: List[str]) -> List[float]:
    parsed = [parse_response(r) for r in responses]
    answers = [_normalize_answer(p['answer']) for p in parsed]
    
    counter = Counter(answers)
    _, maj_count = counter.most_common(1)[0]
    agreements = [
        (counter[a] - 1) / max(1, len(answers) - 1) for a in answers
    ]
    group_agreement = (maj_count - 1) / max(1, len(answers) - 1)
    rewards = []
    for p, agree in zip(parsed, agreements):
        r = 0.0
        if not p['has_cot']:
            rewards.append(-1.5)
            continue
        r += 1.0 * format_score(p['cot'])
        r += 1.0 * length_score(p['cot'], group_agreement)
        r += 0.4 * agree
        
        if len(p['answer']) < 2: r -= 0.5
        rewards.append(r)
    return rewards
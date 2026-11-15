"""Fixed-answer evaluation harness, independent of model inference."""
import json,re
from decimal import Decimal,InvalidOperation

def grade(case,reply):
    raw=reply.strip();kind=case['grading'];expected=case['expected'];semantic=False;format_ok=True
    if kind=='json':
        candidate=raw
        if candidate.startswith('```') and candidate.endswith('```'):
            format_ok=False;lines=candidate.splitlines();candidate='\n'.join(lines[1:-1])
        try:
            actual=json.loads(candidate)
            semantic=json.dumps(actual,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False)
        except (ValueError,TypeError):format_ok=False
    elif kind=='number':
        format_ok=bool(re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',raw))
        if format_ok:
            try:semantic=Decimal(raw).is_finite() and Decimal(raw)==Decimal(str(expected))
            except InvalidOperation:pass
    elif kind=='csv_numbers':
        parts=[v.strip() for v in raw.split(',')];format_ok=all(re.fullmatch(r'[+-]?\d+',v) for v in parts)
        semantic=bool(format_ok) and [int(v) for v in parts]==expected
    elif kind=='csv_words':
        semantic=[v.strip().casefold() for v in raw.split(',')]==[v.casefold() for v in expected]
    elif kind=='exact_text':semantic=raw==expected
    elif kind=='text':
        semantic=raw.casefold().rstrip('.')==expected.casefold().rstrip('.')
        format_ok=raw.casefold()==expected.casefold()
    else:raise ValueError(kind)
    return {'semantic_correct':bool(semantic),'format_ok':bool(format_ok),'passed':bool(semantic and (format_ok or case['category']!='format'))}

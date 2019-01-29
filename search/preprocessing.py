import datetime


def is_blacklisted(word):
    return word in [
        'polizei', 'polizist', 'beamter', 'nr.', 'berlin', 'uhr', 'polizeimeldung',
        'nicht', 'jahr', 'jährige', 'jährig', 'jähriger', 'polizeiliche', 'polizeilich', '2015', '2016',
        '2014', '2017', '2018', 'polizeibeamter', '-', 'u.a.', 'z.b.', 'der', 'die', 'das', 'dem', 'den', 'diese',
        'dieser', 'diesen', 'diesem', 'um', 'für', 'eine', 'ein', 'einer', 'einen', 'einem', 'anderer',
        'andere', 'anderen', 'anders'
    ]


def is_empty(word):
    return word.strip() == ''


def can_parse_date(word):
    try:
        datetime.datetime.strptime(word, '%d.%m.%Y')
        return True
    except ValueError:
        return False


def can_parse_num_int(word):
    try:
        int(word)
        return True
    except ValueError:
        return False


def can_parse_num_float(word):
    try:
        float(word)
        return True
    except ValueError:
        return False


def preprocess_after(doc):
    return [
        word
        for word in doc
        if not is_empty(word)
           and not is_blacklisted(word)
           and not can_parse_date(word)
           and not can_parse_num_int(word)
           and not can_parse_num_float(word)
    ]
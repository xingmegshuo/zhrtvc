import re
import inflect

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

_number_cn = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
_number_level = ['千', '百', '十', '万', '千', '百', '十', '亿', '千', '百', '十', '万', '千', '百', '十', '个']
_zero = _number_cn[0]
_ten_re = re.compile(r'^一十')
_grade_level = {'万', '亿', '个'}
_number_group_re = re.compile(r"([0-9]+)")


def number_digit(num: str):
    outs = []
    for zi in num:
        outs.append(_number_cn[int(zi)])
    return ''.join(outs)


def number_number(num: str):
    x = str(int(num))
    if x == '0':
        return _number_cn[0]
    length = len(x)
    outs = []
    for num, zi in enumerate(x):
        a = _number_cn[int(zi)]
        b = _number_level[len(_number_level) - length + num]
        if a != _zero:
            outs.append(a)
            outs.append(b)
        else:
            if b in _grade_level:
                if outs[-1] != _zero:
                    outs.append(b)
                else:
                    outs[-1] = b
            else:
                if outs[-1] != _zero:
                    outs.append(a)
    out = ''.join(outs[:-1])
    out = _ten_re.sub(r'十', out)
    return out


def number_decimal(num: str):
    z, x = num.split('.')
    z_cn = number_number(z)
    x_cn = number_digit(x)
    return z_cn + '点' + x_cn


def number_convert(text: str):
    parts = _number_group_re.split(text)
    outs = []
    for elem in parts:
        if elem.isdigit():
            if len(elem) <= 9:
                outs.append(number_number(elem))
            else:
                outs.append(number_digit(elem))
        else:
            outs.append(elem)
    return ''.join(outs)


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

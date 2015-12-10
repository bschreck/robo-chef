import re
def phrase2words(phrase):
    return [re.sub('\d', '1', w) for w in re.findall(r"[\w']+|[^\w\s]", phrase, re.UNICODE)]

import textacy


def remove_middle_names(name: str) -> str:
    names = name.split()
    return ' '.join([names[0], names[-1]])


def get_stop_words(nlp):
    custom_set = {
        '$', 'paper', 'propose', 'approach',
        'method', 'i.e.', 'e.g.', '`'
    }
    stop_words = nlp.Defaults.stop_words
    stop_words |= custom_set
    stop_words |= {w.capitalize() for w in stop_words}
    return stop_words

import textacy
from pathlib import Path


def load(src: str = 'data/nlp/topic_model_21.pkl') -> textacy.TopicModel:
    '''
    Load a topic model from disk.

    Arguments:
        src {str} -- Path to topic model file.

    Returns:
        textacy.TopicModel -- A trained topic model.
    '''
    if not Path(src).is_file():
        raise FileNotFoundError('Topic model file does not exist.')
    if Path(src).suffix != '.pkl':
        raise TypeError('File mustu be of type .pkl')
    return textacy.tm.TopicModel.load(src)


def save(model: textacy.TopicModel, src: str) -> None:
    if Path(src).suffix != '.pkl':
        raise TypeError('File must be of type .pkl')
    return model.save(src)

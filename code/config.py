import yaml


class LoadConfig:
    """
    Change key-values in dictionary into properties
    """
    def __init__(self, dictionary):
        """
        :param dictionary: dictionary loaded from configuration
        """
        for key, value in dictionary.items():
            self.__dict__[key] = value


def config():
    """
    :return: configuration of the model
    """
    with open('cfg.yaml', 'r', encoding='utf-8') as file:
        d = file.read()
    dic = yaml.load(d, Loader=yaml.FullLoader)
    cfg = LoadConfig(dic)
    return cfg

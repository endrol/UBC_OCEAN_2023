from model_utils.bsline import UBCModel


def choose_model(name, cfg):
    if name=="bsline":
        return UBCModel(cfg)

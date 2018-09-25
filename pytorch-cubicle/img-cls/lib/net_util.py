import torchvision
import model as extra_models

def get_avail_models():
    '''
    '''
    available_models = torchvision.models

    for name in extra_models.__dict__:
        if name.islower() and not name.startswith("__") and callable(extra_models.__dict__[name]):
            available_models.__dict__[name] = extra_models.__dict__[name]

    available_models_names = sorted(
            name for name in available_models.__dict__
            if name.islower() and not name.startswith("__")
            and callable(available_models.__dict__[name]))

    return available_models, available_models_names 

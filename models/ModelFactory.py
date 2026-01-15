import models


class ModelFactory:
    @staticmethod
    def create_instance(config):
        model_name = config.model['name']
        model_kwargs = config.model
        del model_kwargs['name']

        if model_name not in models.__dict__:
            raise Exception(f"Could not find model: {model_name}")
        model_class = getattr(models, model_name)

        try:
            model = model_class(**model_kwargs)
        except TypeError as e:
            raise TypeError(f"Could not instantiate {model_name} with {model_kwargs}\n{e}")
        return model
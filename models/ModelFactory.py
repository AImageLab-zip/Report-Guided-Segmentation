import models


class ModelFactory:
    @staticmethod
    def print_model_info(model, model_name):
        """Print model architecture and parameter counts."""
        print("\n" + "="*80)
        print(f"Model: {model_name}")
        print("="*80)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("="*80)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*80 + "\n")
    
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
        
        ModelFactory.print_model_info(model, model_name)
        
        return model
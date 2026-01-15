import torch

class OptimizerFactory:
    @staticmethod
    def create_instance(model, config):
        optimizer_name = config.optimizer['name']
        optimizer_kwargs = config.optimizer
        del optimizer_kwargs['name']

        if optimizer_name not in torch.optim.__dict__:
            raise Exception(f"Could not find optimizer: {optimizer_name}")
        optimizer_class = getattr(torch.optim, optimizer_name)

        try:
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        except TypeError as e:
            raise TypeError(f"Could not instantiate {optimizer_name} with {optimizer_kwargs}\n{e}")

        if hasattr(config, 'scheduler'):
            scheduler_name = config.scheduler['name']
            scheduler_kwargs = config.scheduler
            del scheduler_kwargs['name']

            if scheduler_name not in torch.optim.lr_scheduler.__dict__:
                raise Exception(f"Could not find optimizer: {scheduler_name}")
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)

            try:
                scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            except TypeError as e:
                raise TypeError(f"Could not instantiate {scheduler_name} with {scheduler_kwargs}\n{e}")
        else:
            scheduler = None

        return optimizer, scheduler
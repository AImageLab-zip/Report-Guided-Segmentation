import torch

class OptimizerFactory:
    @staticmethod
    def create_instance(model, config, extra_params=None):
        extra_params = [] if extra_params is None else list(extra_params)

        optimizer_kwargs = dict(config.optimizer)
        optimizer_name = optimizer_kwargs.pop("name")

        if optimizer_name not in torch.optim.__dict__:
            raise Exception(f"Could not find optimizer: {optimizer_name}")
        optimizer_class = getattr(torch.optim, optimizer_name)

        params = list(model.parameters()) + extra_params
        optimizer = optimizer_class(params, **optimizer_kwargs)

        if hasattr(config, 'scheduler'):
            scheduler_kwargs = dict(config.scheduler)
            scheduler_name = scheduler_kwargs.pop("name")

            if scheduler_name not in torch.optim.lr_scheduler.__dict__:
                raise Exception(f"Could not find scheduler: {scheduler_name}")
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        else:
            scheduler = None

        return optimizer, scheduler

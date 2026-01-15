import losses
import monai


class LossFactory:
    @staticmethod
    def create_instance(config):
        loss_name = config.loss['name']
        loss_kwargs = config.loss
        del loss_kwargs['name']

        if loss_name in monai.losses.__dict__:
            loss_class = getattr(monai.losses, loss_name)
            print(f'Loading {loss_name} from Monai')
        elif loss_name in losses.__dict__:
            loss_class = getattr(losses, loss_name)
        else:
            raise Exception(f"Could not find loss: {loss_name}")

        try:
            loss = loss_class(**loss_kwargs)
        except TypeError as e:
            raise TypeError(f"Could not instantiate {loss_name} with {loss_kwargs}\n{e}")
        return loss_name, loss
import transforms
import torchio as tio
from torchvision.transforms import Compose

class TransformsFactory:
    @staticmethod
    def create_instance(transforms_config):
        """
        Create a composed transform from a list of transform configurations.

        :param transforms_config: A list of dictionaries containing transform names and parameters.
        :return: A composed transform (tio.Compose) or None if no transforms are specified.
        """
        transform_instances = []
        for transform_cfg in transforms_config:
            name = transform_cfg['name']
            params = transform_cfg.get('params', {})

            try:
                transform_class = getattr(tio.transforms, name)
                #transform_class = getattr(transforms, name)
            except AttributeError:
                raise ValueError(f"Transform '{name}' not found in torchio.transforms")

            try:
                transform_instance = transform_class(**params)
                transform_instances.append(transform_instance)
            except TypeError as e:
                raise TypeError(f"Could not instantiate transform '{name}' with parameters {params}\n{e}")

        if transform_instances:
            return tio.Compose(transform_instances)
        #if transform_instances:
            #return Compose(transform_instances)
        else:
            return None
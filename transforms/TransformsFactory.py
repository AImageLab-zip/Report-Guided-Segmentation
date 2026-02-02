import transforms
import torchio as tio
from torchvision.transforms import Compose
import monai.transforms as mt
from monai.transforms import Compose as MonaiCompose
import inspect

class TransformsFactory:
    @staticmethod
    def create_instance(transforms_config, backend="torchio", default_keys=("image", "label")):
        """
        Create a composed transform from a list of transform configurations.

        :param transforms_config: A list of dictionaries containing transform names and parameters.
        :return: A composed transform (tio.Compose) or None if no transforms are specified.
        """
        if not transforms_config:
                    return None

        backend = (backend or "torchio").lower()
        transform_instances = []

        if backend == "torchio":
             
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
            
        elif backend == "monai":
             for cfg in transforms_config:
                name = cfg["name"]
                params = cfg.get("params", {}) or {}
                keys = cfg.get("keys", None)

                transform_class = getattr(mt, name)

                # inject keys if transform supports it and user didn't provide it
                sig = inspect.signature(transform_class.__init__)
                if "keys" in sig.parameters and "keys" not in params:
                    params = dict(params)
                    params["keys"] = list(keys) if keys is not None else list(default_keys)

                transform_instances.append(transform_class(**params))

             return MonaiCompose(transform_instances)

        else: 
            raise ValueError(f"Unknown backend: {backend}")
             
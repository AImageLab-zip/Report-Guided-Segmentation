import metrics
import monai

# Class to create a dictionary containing the metrics specified in the config file
class MetricsFactory:
    @staticmethod
    def create_instance(config) -> dict:
        metrics_name = config.metrics['name']
        #TODO: Handle possibly metrics parameters
        #metrics_kwargs = config.metrics
        #del metrics_kwargs['name']

        metrics_dict = {}
        for m in metrics_name:
            if m in monai.metrics.__dict__:
                m_class = getattr(monai.metrics, m)
                print(f'Loading {m} from Monai')
            elif m in metrics.__dict__:
                m_class = getattr(metrics, m)
            else:
                raise Exception(f"Could not find metric: {m}")
            try:
                metric = m_class()
                metrics_dict[m] = metric
            except TypeError as e:
                raise TypeError(f"Could not instantiate {m}\n{e}")

        return metrics_dict
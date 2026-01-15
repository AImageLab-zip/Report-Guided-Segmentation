import json
from pathlib import Path


class Config:
    def __init__(self, file_path):
        self.config_path = Path(file_path)
        with self.config_path.open('rt') as file:
            data = json.load(file)

        for key, value in data.items():
            setattr(self, key, value)

    def __repr__(self, ):
        repr_str = ""
        for key, value in self.__dict__.items():
            repr_str += f'{key}: {value}\n'
        return repr_str

    def __str__(self, ):
        return self.__repr__()
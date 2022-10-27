import argparse
import json
import os

class Config():
    def __init__(self, cfg=None, **kwargs):
        self.update(cfg=cfg, **kwargs)
    def update(self, cfg=None, **kwargs):
        if cfg is not None:
            if isinstance(cfg, dict):
                self._add_attributes(cfg)
            elif isinstance(cfg, str):
                assert os.path.isfile(cfg), 'No cfg file : %s'%cfg
                with open(cfg) as f:
                    json_data = json.load(f)
                    self._add_attributes(json_data)
        self._add_attributes(kwargs)
            
    def _add_attributes(self, attr_dict):
        for key in attr_dict:
            setattr(self, key, attr_dict[key])
            
    def __str__(self):
        out = ['[Configuration]']
        for arg in self.__dict__:
            out.append(f'\t{arg}:\t{getattr(self,arg)}')
        return '\n'.join(out)
        
    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

        

def get_config(cfg=None, **kwargs):
    config = Config(cfg=cfg, **kwargs)
    
    parser = argparse.ArgumentParser()
    attrs = config.__dict__
    for key in attrs:
        parser.add_argument('--%s'%key, type=type(attrs[key]), default=attrs[key])
    args = parser.parse_args(namespace=config)
    
    return args
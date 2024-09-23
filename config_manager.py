import os
import argparse
import configparser

class ConfigManager:
    """
        Going to use this bad boy to determine inference server address so i don't have to manually change it in gen_features
    """
    def __init__(self, model, config_file="config.cfg"):
        self.model = model
        self.config_file = config_file
        self.config = configparser.ConfigParser()

    @staticmethod
    def get_all_keys(config_file="config.cfg"):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config.sections()
    
    @staticmethod
    def get_model_args(model, config_file="config.cfg"):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config[model].__dict__()
    
    def write_args(self, args):
        self.config[str(self.model)] = {key: str(val) for key, val in vars(args).items()}
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)

    def read_args(self):
        self.config.read(self.config_file)
        args = self.config[self.model]
        return argparse.Namespace(**args)

    def read_single_arg(self, arg_name):
        self.config.read(self.config_file)
        return self.config[self.model].get(arg_name, None)

    def config_model_exists(self):
        if ConfigManager.config_exists():
            self.config.read(self.config_file)
            if self.config.has_section(self.model):
                return True
        return False  
      
    @staticmethod
    def config_exists(config_file="config.cfg"):
        if os.path.exists(config_file):
            return True
        return False
"""
Support functions for arg parsing to config
"""

import argparse
import os
import yaml

class Nestedspace(argparse.Namespace):
    """
    Define a nested namespace for dealing with nested args.
    from https://stackoverflow.com/questions/18668227/argparse-subcommands-with-nested-namespaces
    """
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

def nestedspace_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Nestedspace:
  """
  Construct a nestedspace.
  adapted from https://matthewpburruss.com/post/yaml/
  """
  return Nestedspace(**loader.construct_mapping(node))

def get_nestedspace_loader():
  """
  Add nestedspace constructors to PyYAML loader.
  adapted from https://matthewpburruss.com/post/yaml/
  """
  loader = yaml.SafeLoader
  loader.add_constructor("tag:yaml.org,2002:python/object:config.config.Nestedspace", nestedspace_constructor)
  return loader

def config_to_yaml(config, save_path, save_name=None):
    """Save nestedspace config to yaml file."""
    if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        yaml_file_name = os.path.join(save_path,'config.yaml')
    else:
        yaml_file_name = os.path.join(save_path,f"{save_name}.yaml")
    print(f"--> save config as yaml at {yaml_file_name}")
    with open(yaml_file_name, "w", encoding = "utf-8") as yaml_file:
        dump = yaml.dump(config, default_flow_style = False, allow_unicode = True, encoding = None)
        yaml_file.write(dump)
        
    return yaml_file_name

def yaml_to_config(yaml_path, new_log_dir, new_run_name):
    """Load yaml into nestedspace config"""
    with open(yaml_path, "r") as yaml_file:
        saved_config = yaml.load(yaml_file, Loader=get_nestedspace_loader())
        saved_config.log_dir = new_log_dir # Modify the log path to a new directory so we don't overwrite anything
        saved_config.new_run_name = new_run_name # Modify the log path to a new directory so we don't overwrite anything
    return saved_config

def check_for_unknown_args(config,args_to_check):
    """Check if user input any unknown args after completing all parsing"""
    unknown_args = []
    for arg in args_to_check:
        if arg not in config: 
            if '.' in arg:
                config_check = config
                for arg_element in arg.split('.'):
                    if arg_element in config_check:
                        config_check = getattr(config_check, arg_element)
                    else:
                        unknown_args += [arg]
                        break
            else: 
                unknown_args += [arg]
    return unknown_args
    
def none_or_str(value):
    """Convert arg from a string to None"""
    if value in ['None', 'none', None]:
        return None
    return value

def str_to_bool(value):
    """Convert arg from a string to bool"""
    if str(value) in ['1','True','true','T','t']: return True
    else: return False

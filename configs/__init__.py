
__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import os
import sys
import json
import yaml

from utils.logger import Logger
logger = Logger.get_logger()


class Config(object):
    def __init__(self, cfg_path=None, cfg=None):
        if cfg_path is None and cfg is None:
            raise ValueError("the path or cfg must be not None.")
        if cfg is None:
            with open(cfg_path, "r", encoding='utf-8') as f:
                cfg_str = f.read()
                self._cfg = yaml.load(cfg_str, Loader=yaml.FullLoader)
        else:
            self._cfg = cfg
        # scan the configs, then set attribute
        self.load_dict_as_attributes(self._cfg)

    def load_dict_as_attributes(self, cfg):
        for k, v in cfg.items():
            if isinstance(v, dict):
                sub_obj = Config(cfg=v)
                self.__setattr__(k, sub_obj)
            else:
                self.__setattr__(k, v)

    def __str__(self):
        return str(self.__dict__)


class Configer(object):

    def __init__(self, args_parser=None, configs=None, config_dict=None):
        if config_dict is not None:
            self.params_root = config_dict

        elif configs is not None:
            if not os.path.exists(configs):
                logger.write('Json Path:{} not exists!'.format(configs))
                exit(0)

            json_stream = open(configs, 'r')
            self.params_root = json.load(json_stream)
            json_stream.close()

        elif args_parser is not None:
            self.args_dict = args_parser.__dict__
            self.params_root = None

            if not os.path.exists(args_parser.configs):
                print('Json Path:{} not exists!'.format(args_parser.configs))
                exit(1)

            json_stream = open(args_parser.configs, 'r')
            self.params_root = json.load(json_stream)
            json_stream.close()

            for key, value in self.args_dict.items():
                if not self.exists(*key.split(':')):
                    self.add(key.split(':'), value)
                elif value is not None:
                    self.update(key.split(':'), value)

    def _get_caller(self):
        filename = os.path.basename(sys._getframe().f_back.f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_back.f_lineno
        prefix = '{}, {}'.format(filename, lineno)
        return prefix

    def get(self, *key):
        if len(key) == 0:
            return self.params_root

        elif len(key) == 1:
            if key[0] in self.params_root:
                return self.params_root[key[0]]
            else:
                logger.write('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        elif len(key) == 2:
            if key[0] in self.params_root and key[1] in self.params_root[key[0]]:
                return self.params_root[key[0]][key[1]]
            else:
                logger.write('{} KeyError: {}.'.format(self._get_caller(), key))
                exit(1)

        else:
            logger.write('{} KeyError: {}.'.format(self._get_caller(), key))
            exit(1)

    def exists(self, *key):
        if len(key) == 1 and key[0] in self.params_root:
            return True

        if len(key) == 2 and (key[0] in self.params_root and key[1] in self.params_root[key[0]]):
            return True

        return False

    def add(self, key_tuple, value):
        if self.exists(*key_tuple):
            logger.write('{} Key: {} existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        if len(key_tuple) == 1:
            self.params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            if key_tuple[0] not in self.params_root:
                self.params_root[key_tuple[0]] = dict()

            self.params_root[key_tuple[0]][key_tuple[1]] = value

        else:
            logger.write('{} KeyError: {}.'.format(self._get_caller(), key_tuple))
            exit(1)

    def update(self, key_tuple, value):
        if not self.exists(*key_tuple):
            logger.write('{} Key: {} not existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

        if len(key_tuple) == 1 and not isinstance(self.params_root[key_tuple[0]], dict):
            self.params_root[key_tuple[0]] = value

        elif len(key_tuple) == 2:
            self.params_root[key_tuple[0]][key_tuple[1]] = value

        else:
            logger.write('{} Key: {} not existed!!!'.format(self._get_caller(), key_tuple))
            exit(1)

    def resume(self, config_dict):
        self.params_root = config_dict

    def plus_one(self, *key):
        if not self.exists(*key):
            logger.write('{} Key: {} not existed!!!'.format(self._get_caller(), key))
            exit(1)

        if len(key) == 1 and not isinstance(self.params_root[key[0]], dict):
            self.params_root[key[0]] += 1

        elif len(key) == 2:
            self.params_root[key[0]][key[1]] += 1

        else:
            logger.write('{} KeyError: {} !!!'.format(self._get_caller(), key))
            exit(1)

    def to_dict(self):
        return self.params_root
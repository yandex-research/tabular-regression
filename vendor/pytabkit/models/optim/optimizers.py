# Vendored pytabkit snapshot, adapted for the regression benchmark.
# Source: https://github.com/dholzmueller/pytabkit
# License: Apache-2.0; see LICENSE.txt. Local changes: see VENDOR.md.

import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Optional, Dict, Any, Set, DefaultDict, Iterable
from pathlib import Path
import sys

import torch
import torch.optim as optim
from torch.optim.optimizer import required, StateDict

from pytabkit.models.training.coord import HyperparamManager
from pytabkit.models.optim.scheduling_adam import SchedulingAdam

from vendor.muon import SingleDeviceMuonWithAuxAdam


class OptimizerBase(torch.optim.Optimizer):
    def __init__(self, opt, hyper_mappings, hp_manager: HyperparamManager):
        self.hp_manager = hp_manager
        self.hyper_getters = {}
        self.n_groups = len(opt.param_groups)
        for names, opt_name, defaults in hyper_mappings:
            if isinstance(names, str):
                names = (names,)
                defaults = (defaults,)
            for name, default in zip(names, defaults):
                self.hyper_getters[name] = [self.hp_manager.register_hyper(name, group['params'][0].context.scope,
                                                                           default=default) for group in opt.param_groups]
        super().__init__(opt.param_groups, defaults={})
        self.hyper_mappings = hyper_mappings
        self.opt = opt

    def get_hyper_values(self, name, i, use_hyper_factor=True):
        value = self.hyper_getters[name][i]()
        param = self.opt.param_groups[i]['params'][0]  # should only be one param
        if use_hyper_factor and name in param.hyper_factors:
            value *= param.hyper_factors[name]
        return value

    def step(self, closure=None, loss: Optional[torch.Tensor] = None):
        unhandled_mappings = []
        for names, opt_name, defaults in self.hyper_mappings:
            if opt_name is None:
                unhandled_mappings.append((names, opt_name, defaults))
                continue

            if isinstance(names, tuple):
                for i, group in enumerate(self.opt.param_groups):
                    group[opt_name] = tuple(self.get_hyper_values(name, i) for name in names)
            elif isinstance(names, str):
                for i, group in enumerate(self.opt.param_groups):
                    group[opt_name] = self.get_hyper_values(names, i)
            else:
                raise RuntimeError('Could not understand mapping key {}'.format(names))

        for names, opt_name, defaults in unhandled_mappings:
            if names == 'wd':
                with torch.no_grad():
                    for i, group in enumerate(self.opt.param_groups):
                        wd = self.get_hyper_values('wd', i)
                        lr = self.get_hyper_values('lr', i)
                        if wd != 0.0:
                            for p in group['params']:
                                p.mul_(1.0 - wd * lr * p.hyper_factors.get('wd', 1.0) * p.hyper_factors.get('lr', 1.0))

            else:
                raise RuntimeError('Could not understand mapping {}'.format((names, opt_name, defaults)))

        self._opt_step_with_loss(loss)

    def train(self):
        if hasattr(self.opt, 'train') and callable(self.opt.train):
            # print('opt train')
            self.opt.train()

    def eval(self):
        if hasattr(self.opt, 'eval') and callable(self.opt.eval):
            # print('opt eval')
            self.opt.eval()

    def _opt_step_with_loss(self, loss: Optional[torch.Tensor]):
        self.opt.step()

    def __getstate__(self) -> Dict[str, Any]:
        # override the pickling method since otherwise self.opt is not restored
        return {'__dict__': self.__dict__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # override the pickling method since otherwise self.opt is not restored
        self.__dict__ = state['__dict__']


def _is_realmlp_muon_param(param, config: Dict[str, Any]) -> bool:
    """
    Use some internal property of pytabkit nn models to easily distinguish models

    Example model layer structure from a debug run with a print

    DEBUG: /plr/weight_1
    DEBUG: /plr/weight_2
    DEBUG: /plr/bias_1
    DEBUG: /plr/bias_2
    DEBUG: /net/first_layer/layer-0/scale
    DEBUG: /net/first_layer/layer-0/weight
    DEBUG: /net/first_layer/layer-0/bias
    DEBUG: /net/first_layer/layer-0/act
    DEBUG: /net/middle_layer/layer-1/weight
    DEBUG: /net/middle_layer/layer-1/bias
    DEBUG: /net/middle_layer/layer-1/act
    DEBUG: /net/middle_layer/layer-2/weight
    DEBUG: /net/middle_layer/layer-2/bias
    DEBUG: /net/middle_layer/layer-2/act
    DEBUG: /net/last_layer/layer-3/weight
    DEBUG: /net/last_layer/layer-3/bias

    By default we do not apply muon to the last layer
    and to the embedding layers (even the NLinears there)
    """
    scope = str(param.context.scope)
    # print('DEBUG:', scope)
    include_plr = config.get('muon_include_plr', False)

    if scope.endswith('/weight') and '/net/' in scope:
        if '/net/last_layer/' in scope:
            return False
        return True

    if include_plr and '/plr/' in scope and '/weight_' in scope:
        return True

    return False


class MuonOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager: HyperparamManager, **config):
        self.config = config
        muon_param_groups = []
        for group in param_groups:
            p = group['params'][0]
            use_muon = _is_realmlp_muon_param(p, config)
            if use_muon:
                muon_param_groups.append({
                    'params': group['params'],
                    'lr': 1e-3,
                    'weight_decay': 0.0,
                    'use_muon': True
                })
            else:
                muon_param_groups.append({
                    'params': group['params'],
                    'lr': 1e-3,
                    'betas': (0.9, 0.999),
                    'eps': 1e-8,
                    'weight_decay': 0.0,
                    'use_muon': False
                })

        super().__init__(
            SingleDeviceMuonWithAuxAdam(muon_param_groups),
            hyper_mappings=[
                ('muon_lr', None, 1e-2), ('muon_mom', None, 0.95),
                ('lr', 'lr', 1e-3), ('mom', None, 0.95),
                ('sq_mom', None, 0.999), ('opt_eps', None, 1e-8), ('wd', None, 0.0)
            ],
            hp_manager=hp_manager
        )

    def step(self, closure=None, loss: Optional[torch.Tensor] = None):
        for i, group in enumerate(self.opt.param_groups):
            if group['use_muon']:
                group['lr'] = self.get_hyper_values('muon_lr', i)
                group['momentum'] = self.get_hyper_values('muon_mom', i)
                # print(f"DEBUG: muon lr = {group['lr']}")
            else:
                group['lr'] = self.get_hyper_values('lr', i)
                group['betas'] = (self.get_hyper_values('mom', i), self.get_hyper_values('sq_mom', i))
                group['eps'] = self.get_hyper_values('opt_eps', i)
                # print(f"DEBUG: adam lr = {group['lr']}")

        # weight decay set to 0 in the optimizers, and applied here with the hyper_factors
        with torch.no_grad():
            for i, group in enumerate(self.opt.param_groups):
                wd = self.get_hyper_values('wd', i)
                if wd != 0.0:
                    for p in group['params']:
                        lr = self.get_hyper_values('muon_lr' if group['use_muon'] else 'lr', i)
                        p.mul_(1.0 - wd * lr * p.hyper_factors.get('wd', 1.0) * p.hyper_factors.get('lr', 1.0))

        self._opt_step_with_loss(loss)


class AdamOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager, **config):
        super().__init__(optim.Adam(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class SchedulingAdamOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager, **config):
        super().__init__(SchedulingAdam(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class AMSGradOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager, **config):
        super().__init__(optim.Adam(param_groups, amsgrad=True),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class AdamaxOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager, **config):
        super().__init__(optim.Adamax(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class SGDOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager, **config):
        super().__init__(optim.SGD(param_groups), hyper_mappings=[('lr', 'lr', 1e-3), ('mom', 'momentum', 0.0),
                                                                  ('wd', None, 0.0)],
                         hp_manager=hp_manager)

class SFAdamOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager: HyperparamManager, **config):
        from schedulefree import AdamWScheduleFree
        super().__init__(AdamWScheduleFree(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0),
                                         ('weight_decay', 'weight_decay', 0.0),
                                         ('warmup_steps', 'warmup_steps', 0)],
                         hp_manager=hp_manager)


class MoMoAdamOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager: HyperparamManager, **config):
        from momo import MomoAdam
        super().__init__(MomoAdam(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)

    def _opt_step_with_loss(self, loss: Optional[torch.Tensor]):
        self.opt.step(loss=loss)


class AdoptOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager: HyperparamManager, **config):
        from .adopt import ADOPT
        super().__init__(ADOPT(param_groups, decoupled=True),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


def get_opt_class(opt_name):
    if opt_name == 'adam':
        return AdamOptimizer
    elif opt_name == 'adamax':
        return AdamaxOptimizer
    elif opt_name == 'sgd':
        return SGDOptimizer
    elif opt_name == 'amsgrad':
        return AMSGradOptimizer
    elif opt_name == 'sched_adam':
        return SchedulingAdamOptimizer
    elif opt_name == 'sfadam':
        return SFAdamOptimizer
    elif opt_name == 'momoadam':
        return MoMoAdamOptimizer
    elif opt_name == 'adopt':
        return AdoptOptimizer
    elif opt_name == 'muon':
        return MuonOptimizer
    else:
        raise ValueError(f'Unknown optimizer "{opt_name}"')

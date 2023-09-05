import logging
from contextlib import contextmanager
import sys
from typing import Any

_MPU_MODULE_NAME = None
_NAME = 0
_topo_entrypoint = {}

__is_optimus_parallel_unit__ = True
__mpu_name__ = None
__mpu__ = None


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def _find_mpu_module():
    global _MPU_MODULE_NAME
    if _MPU_MODULE_NAME is not None:
        return

    for k, v in sys.modules.items():
        if "mpu" in k and hasattr(v, "__is_optimus_mpu__") and v.__is_optimus_mpu__:
            _MPU_MODULE_NAME = k
            return
    raise RuntimeError("Cannot find mpu module")


def _del_mpu_module():
    assert _MPU_MODULE_NAME is not None, "MPU module name must be found"
    module_name = _MPU_MODULE_NAME
    to_delete = [module_name]

    for mod in list(sys.modules):
        if mod.startswith(module_name + "."):
            to_delete.append(mod)

    for mod in to_delete:
        del sys.modules[mod]


def initialize_model_parallel(
    model_parallel_size, topology=None, fp32_allreduce=False, model_name=None
):
    # _find_mpu_module()
    global _topo_entrypoint, __mpu__, __mpu_name__, _NAME
    if model_name is None:
        logging.warning(
            "No model name is provided, using default name: {}".format(_NAME)
        )
        model_name = str(_NAME)
        _NAME += 1

    try:
        import optimus._mpu
    except ImportError:
        raise ImportError("Must INSTALL optimus first")

    _find_mpu_module()
    # _set_property()

    optimus._mpu.initialize_model_parallel(
        model_parallel_size, topology, fp32_allreduce
    )

    # regiter mpu
    _topo_entrypoint[model_name] = optimus._mpu
    __mpu__ = optimus._mpu
    __mpu_name__ = model_name

    _del_mpu_module()

def _init_mpu():
    global __mpu__
    try:
        import optimus._mpu
    except ImportError:
        raise ImportError("Must INSTALL optimus first")

    _find_mpu_module()
    __mpu__ = optimus._mpu
    _del_mpu_module()


def _select_mpu(name):
    if not isinstance(name, str):
        name = str(name)
    global __mpu__, __mpu_name__
    if name not in _topo_entrypoint:
        raise ValueError("Unknown model parallel name: {}".format(name))
    __mpu__ = _topo_entrypoint[name]
    __mpu_name__ = name


class MPUModule(metaclass=SingletonMeta):
    # ===========================================
    #
    # Modules (cross entropy, data, initialize, layers, lora, mappings, random, utils)
    #
    # ===========================================
    @property
    def cross_entropy(self):
        return __mpu__.cross_entropy

    @property
    def data(self):
        return __mpu__.data

    @property
    def initialize(self):
        return __mpu__.initialize

    @property
    def layers(self):
        return __mpu__.layers

    @property
    def lora(self):
        return __mpu__.lora

    @property
    def mappings(self):
        return __mpu__.mappings

    @property
    def random(self):
        return __mpu__.random

    @property
    def utils(self):
        return __mpu__.utils
    # ===========================================
    #
    # Reflected functions
    #
    # ===========================================

    def __getattr__(self, __name: str) -> Any:
        return getattr(__mpu__, __name)

    @staticmethod
    def vocab_parallel_cross_entropy(*args, **kwargs):
        return __mpu__.vocab_parallel_cross_entropy(*args, **kwargs)

    @staticmethod
    def broadcast_data(*args, **kwargs):
        return __mpu__.broadcast_data(*args, **kwargs)

    @staticmethod
    def is_unitialized(*args, **kwargs):
        return __mpu__.is_unitialized(*args, **kwargs)

    @staticmethod
    def destroy_model_parallel(*args, **kwargs):
        return __mpu__.destroy_model_parallel(*args, **kwargs)

    @staticmethod
    def get_data_parallel_group(*args, **kwargs):
        return __mpu__.get_data_parallel_group(*args, **kwargs)

    @staticmethod
    def get_data_parallel_rank(*args, **kwargs):
        return __mpu__.get_data_parallel_rank(*args, **kwargs)

    @staticmethod
    def get_data_parallel_world_size(*args, **kwargs):
        return __mpu__.get_data_parallel_world_size(*args, **kwargs)

    @staticmethod
    def get_model_parallel_group(*args, **kwargs):
        return __mpu__.get_model_parallel_group(*args, **kwargs)

    @staticmethod
    def get_model_parallel_rank(*args, **kwargs):
        return __mpu__.get_model_parallel_rank(*args, **kwargs)

    @staticmethod
    def set_model_parallel_rank(*args, **kwargs):
        return __mpu__.set_model_parallel_rank(*args, **kwargs)

    @staticmethod
    def get_model_parallel_src_rank(*args, **kwargs):
        return __mpu__.get_model_parallel_src_rank(*args, **kwargs)

    @staticmethod
    def get_data_parallel_src_rank(*args, **kwargs):
        return __mpu__.get_data_parallel_src_rank(*args, **kwargs)

    @staticmethod
    def get_model_parallel_world_size(*args, **kwargs):
        return __mpu__.get_model_parallel_world_size(*args, **kwargs)

    @staticmethod
    def set_model_parallel_world_size(*args, **kwargs):
        return __mpu__.set_model_parallel_world_size(*args, **kwargs)

    @staticmethod
    def get_topology(*args, **kwargs):
        return __mpu__.get_topology(*args, **kwargs)

    @staticmethod
    def get_pipe_parallel_group(*args, **kwargs):
        return __mpu__.get_pipe_parallel_group(*args, **kwargs)

    @staticmethod
    def get_pipe_parallel_rank(*args, **kwargs):
        return __mpu__.get_pipe_parallel_rank(*args, **kwargs)

    @staticmethod
    def get_pipe_parallel_world_size(*args, **kwargs):
        return __mpu__.get_pipe_parallel_world_size(*args, **kwargs)

    @staticmethod
    def get_io_parallel_group(*args, **kwargs):
        return __mpu__.get_io_parallel_group(*args, **kwargs)

    @staticmethod
    def model_parallel_is_initialized(*args, **kwargs):
        return __mpu__.model_parallel_is_initialized(*args, **kwargs)

    @staticmethod
    def ColumnParallelLinear(*args, **kwargs):
        return __mpu__.ColumnParallelLinear(*args, **kwargs)

    @staticmethod
    def RowParallelLinear(*args, **kwargs):
        return __mpu__.RowParallelLinear(*args, **kwargs)

    @staticmethod
    def VocabParallelEmbedding(*args, **kwargs):
        return __mpu__.VocabParallelEmbedding(*args, **kwargs)

    @staticmethod
    def ParallelRelativePositionBias(*args, **kwargs):
        return __mpu__.ParallelRelativePositionBias(*args, **kwargs)

    @staticmethod
    def copy_to_model_parallel_region(*args, **kwargs):
        return __mpu__.copy_to_model_parallel_region(*args, **kwargs)

    @staticmethod
    def gather_from_model_parallel_region(*args, **kwargs):
        return __mpu__.gather_from_model_parallel_region(*args, **kwargs)

    @staticmethod
    def reduce_from_model_parallel_region(*args, **kwargs):
        return __mpu__.reduce_from_model_parallel_region(*args, **kwargs)

    @staticmethod
    def scatter_to_model_parallel_region(*args, **kwargs):
        return __mpu__.scatter_to_model_parallel_region(*args, **kwargs)

    @staticmethod
    def checkpoint(*args, **kwargs):
        return __mpu__.checkpoint(*args, **kwargs)

    @staticmethod
    def get_cuda_rng_tracker(*args, **kwargs):
        return __mpu__.get_cuda_rng_tracker(*args, **kwargs)

    @staticmethod
    def model_parallel_cuda_manual_seed(*args, **kwargs):
        return __mpu__.model_parallel_cuda_manual_seed(*args, **kwargs)

    @staticmethod
    def divide(*args, **kwargs):
        return __mpu__.divide(*args, **kwargs)

    @staticmethod
    def split_tensor_along_last_dim(*args, **kwargs):
        return __mpu__.split_tensor_along_last_dim(*args, **kwargs)

    # ===========================================
    #
    # Self Owned functions
    #
    # ===========================================

    @staticmethod
    def initialize_model_parallel(*args, **kwargs):
        return initialize_model_parallel(*args, **kwargs)

    @staticmethod
    @contextmanager
    def scope(name):
        before = __mpu_name__
        _select_mpu(name)
        yield
        _select_mpu(before)

    @staticmethod
    def __call__(name):
        _select_mpu(name)

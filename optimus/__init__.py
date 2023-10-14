from .fused_kernels import load_fused_kernels
from .parallel_unit import MPUModule, _init_mpu
import sys

try:
    import optimus
except ImportError:
    raise ImportError("Please install optimus first.")

_init_mpu()
mpu = MPUModule()

sys.modules["optimus.mpu"] = mpu

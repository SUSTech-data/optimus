try:
    import optimus
except ImportError:
    raise ImportError("Please install optimus first.")

import sys
from .parallel_unit import MPUModule, _init_mpu

_init_mpu()
sys.modules["optimus.mpu"] = MPUModule()

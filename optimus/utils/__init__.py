import gc
import torch

def release_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def move_engine(engine, dst="cpu"):
    # not support zero3 now, but not hard
    try:
        import tree
    except ImportError:
        raise ImportError("Please install dm-tree by `pip install dm-tree` first")
    sd = engine.optimizer.state_dict()
    release_cuda()
    gc.disable()
    _OBJs = [t for t in gc.get_objects() if isinstance(t, torch.Tensor)]

    def move_tensor_cpu(x):
        if isinstance(x, torch.Tensor):
            storage = x.storage()
            for obj in _OBJs:
                if obj.storage()._cdata == storage._cdata:
                    obj.data = obj.data.to(dst)
            x.data = x.data.to(dst)

            if x.requires_grad:
                if isinstance(x.grad, torch.Tensor):
                    x.grad.data = x.grad.data.to(dst)

    tree.map_structure(move_tensor_cpu, sd)
    gc.enable()
    release_cuda()

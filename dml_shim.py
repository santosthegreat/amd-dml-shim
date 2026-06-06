import torch_directml
import torch
import torch.cuda

_dml = torch_directml.device(0)
_name = torch_directml.device_name(0)

def _noop(*a, **kw): pass
def _props(x=0): return type('P', (), {'name': _name, 'total_memory': 12*1024**3, 'major': 8, 'minor': 0, 'multi_processor_count': 36})()

torch._C._cuda_init = _noop
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
torch.cuda.current_device = lambda: 0
torch.cuda.is_initialized = lambda: True
torch.cuda._lazy_init = _noop
torch.cuda.get_device_name = lambda x=0: _name
torch.cuda.get_device_properties = _props

# patch internal module too
import torch.c




eof

#!/bin/bash
TORCH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TRANS=$(python3 -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
echo "Patching torch at $TORCH"
echo "Patching transformers at $TRANS"
sed -i 's/torch._C._cuda_init()/pass  # patched for DML/' $TORCH/cuda/__init__.py
sed -i 's/torch._C._cuda_setDevice(device)/pass  # patched for DML/' $TORCH/cuda/__init__.py
sed -i 's/if device < 0 or device >= device_count():/if False:/' $TORCH/cuda/__init__.py
sed -i 's/capturing = torch.cuda.is_current_stream_capturing()/capturing = False  # patched for DML/' $TORCH/optim/optimizer.py
sed -i 's/use_gather_object=self.args.eval_use_gather_object/use_gather_object=getattr(self.args, "eval_use_gather_object", False)/' $TRANS/trainer.py
sed -i 's/if self.args.use_liger_kernel:/if getattr(self.args, "use_liger_kernel", False):/' $TRANS/trainer.py
sed -i 's/if args.eval_on_start:/if getattr(args, "eval_on_start", False):/' $TRANS/trainer.py
sed -i 's/self\.args\.average_tokens_across_devices/getattr(self.args, "average_tokens_across_devices", False)/g' $TRANS/trainer.py
sed -i 's/self\.args\.torch_empty_cache_steps is not None/getattr(self.args, "torch_empty_cache_steps", None) is not None/g' $TRANS/trainer.py
sed -i 's/self\.args\.torch_empty_cache_steps/getattr(self.args, "torch_empty_cache_steps", None)/g' $TRANS/trainer.py
sed -i 's/if self.args.optim in \[OptimizerNames.LOMO, OptimizerNames.ADALOMO\]/if False/' $TRANS/trainer.py
python3 - << 'PYEOF'
import os, transformers
f = os.path.dirname(transformers.__file__) + '/training_args.py'
c = open(f).read()
old = '                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n                # Sometimes the line in the postinit has not been run before we end up here, so just checking we\'re not at\n                # the default value.\n                self._n_gpu = torch.cuda.device_count()\n                if device.type == "cuda":\n                    torch.cuda.set_device(device)'
new = '                device = torch.device("cpu")\n                self._n_gpu = 0'
if old in c:
    open(f,'w').write(c.replace(old,new))
    print("training_args.py patched OK")
PYEOF
echo "Done."

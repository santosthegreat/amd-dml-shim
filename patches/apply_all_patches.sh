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
sed -i 's/rng_states\["cuda"\] = torch.cuda.random.get_rng_state()/rng_states["cuda"] = []  # patched for DML/' $TRANS/trainer.py
sed -i 's/rng_states\["cuda"\] = torch.cuda.random.get_rng_state_all()/rng_states["cuda"] = []  # patched for DML/' $TRANS/trainer.py
sed -i 's/torch.cuda.random.set_rng_state(rng_states\["cuda"\])/pass  # patched for DML/' $TRANS/trainer.py
sed -i 's/torch.cuda.random.set_rng_state_all(rng_states\["cuda"\])/pass  # patched for DML/' $TRANS/trainer.py
sed -i 's/args\.include_for_metrics/getattr(args, "include_for_metrics", getattr(args, "include_inputs_for_metrics", []) or [])/g' $TRANS/trainer.py
sed -i 's/adapters_weights = safe_load_file(filename, device=device)/adapters_weights = safe_load_file(filename, device="cpu")/' ~/.local/lib/python3.10/site-packages/peft/utils/save_and_load.py 2>/dev/null || true
sed -i 's/result\[k\] = f.get_tensor(k)/result[k] = f.get_tensor(k).to("cpu")/' ~/.local/lib/python3.10/site-packages/safetensors/torch.py 2>/dev/null || true
echo "Done."

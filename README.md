Open the anaconda prompt, navigate to the chatbot.py directory and run the following line:

python chatbot.py -batch_size 32

You will receive the following error:

CUDA available: False
CUDA device count: 0
['C:\\Users\\adams\\Desktop\\LLM', 'C:\\Users\\adams\\anaconda3\\python310.zip', 'C:\\Users\\adams\\anaconda3\\DLLs', 'C:\\Users\\adams\\anaconda3\\lib', 'C:\\Users\\adams\\anaconda3', 'C:\\Users\\adams\\anaconda3\\lib\\site-packages', 'C:\\Users\\adams\\anaconda3\\lib\\site-packages\\win32', 'C:\\Users\\adams\\anaconda3\\lib\\site-packages\\win32\\lib', 'C:\\Users\\adams\\anaconda3\\lib\\site-packages\\Pythonwin']
batch size: 32
cpu
Loading model parameters...
Traceback (most recent call last):
  File "C:\Users\adams\Desktop\LLM\chatbot.py", line 189, in <module>
    model = torch.load(f, map_location=torch.device('cpu'))
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 1028, in load
    return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 1246, in _legacy_load
    magic_number = pickle_module.load(f, **pickle_load_args)
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\storage.py", line 337, in _load_from_bytes
    return torch.load(io.BytesIO(b))
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 1028, in load
    return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 1256, in _legacy_load
    result = unpickler.load()
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 1193, in persistent_load
    wrap_storage=restore_location(obj, location),
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 381, in default_restore_location
    result = fn(storage, location)
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 274, in _cuda_deserialize
    device = validate_cuda_device(location)
  File "C:\Users\adams\anaconda3\lib\site-packages\torch\serialization.py", line 258, in validate_cuda_device
    raise RuntimeError('Attempting to deserialize object on a CUDA '
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.

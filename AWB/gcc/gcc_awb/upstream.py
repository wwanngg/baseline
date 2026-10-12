import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
PROJECT = Path(__file__).resolve().parents[1]
AWB_ROOT = PROJECT.parent
UPSTREAM = PROJECT / 'third_party/GCC-official'
DEFAULT_WEIGHTS = PROJECT / 'weights'
MODEL_NAMES = ('nus8', 'gehler')
WEIGHT_FILES = {'text_encoder/model.safetensors': 1361596304, 'unet/diffusion_pytorch_model.safetensors': 3463784120, 'vae/diffusion_pytorch_model.safetensors': 334643268}
REQUIRED = ('model_index.json', 'scheduler/scheduler_config.json', 'text_encoder/config.json', 'tokenizer/merges.txt', 'tokenizer/vocab.json', 'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json', 'unet/config.json', 'vae/config.json', 'feature_extractor/preprocessor_config.json', *WEIGHT_FILES)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def weights_path(weights_dir, model):
    if model not in MODEL_NAMES:
        raise ValueError(f'Unknown GCC checkpoint: {model}')
    root = (Path(weights_dir) / f'gcc_train_on_{model}').resolve()
    for relative in REQUIRED:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing local GCC component: {path}. No automatic download is attempted.')
        if relative in WEIGHT_FILES and path.stat().st_size != WEIGHT_FILES[relative]:
            raise ValueError(f'Incomplete or unexpected weight size: {path}')
    return root

def modules():
    names = ('_s24_gcc_helpers', '_s24_gcc_evaluation', '_s24_gcc_pipeline')
    if all((name in sys.modules for name in names)):
        return tuple((sys.modules[name] for name in names))
    for relative in ('utils/pipeline_utils.py', 'utils/evaluation_utils.py', 'pipelines/pipeline.py'):
        if not (UPSTREAM / relative).is_file():
            raise FileNotFoundError(f'Missing original GCC source: {UPSTREAM / relative}')
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for name, relative in zip(names[:2], ('utils/pipeline_utils.py', 'utils/evaluation_utils.py')):
            spec = importlib.util.spec_from_file_location(name, UPSTREAM / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        path = UPSTREAM / 'pipelines/pipeline.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        changed = 0
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'utils':
                if [alias.name for alias in node.names] != ['apply_laplacian_highpass']:
                    raise ImportError('Unexpected GCC utils import; review upstream changes')
                node.module = names[0]
                changed += 1
        if changed != 1:
            raise ImportError('Cannot isolate the original GCC pipeline import')
        module = types.ModuleType(names[2])
        module.__file__ = str(path)
        sys.modules[names[2]] = module
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    finally:
        sys.dont_write_bytecode = previous
    return tuple((sys.modules[name] for name in names))

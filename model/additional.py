import os
import math
import torch
import random
import numpy as np
import re
import youtokentome as yttm
from typing import List


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def maybe_script(fn: callable) -> callable:
    """
    Applys torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script.
    Took from yandex-research repository: https://github.com/yandex-research/RuLeanALBERT/blob/main/src/modules/functional.py
    May be useful for TPU training with pl
    """
    using_tpu = bool(os.environ.get("TPU_NAME"))
    should_script = int(os.environ.get("USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def clear_text(sent: str) -> str:
    wo_prep = re.sub(r'[^\w\s]', '', sent)
    return wo_prep.lower()


def load_tokenizer(tokenizer_path: str) -> yttm.BPE:
    if tokenizer_path.split('.')[-1] != 'tok':
        raise TypeError('Invalid tokenizer type. Only ".tok" files is allowed')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("No such file in directory.")
    tokenizer = yttm.BPE(model=tokenizer_path)
    return tokenizer


def tokenize(
        sentence: str,
        tokenizer: yttm.BPE
) -> List[int]:
    ans = tokenizer.encode(
        sentences=sentence,
        output_type=yttm.OutputType.ID,
        bos=True, eos=True
    )

    return ans

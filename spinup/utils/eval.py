import numpy as np
import torch
import torch.nn as nn
import os



def eval(pi, env, max_ep_len, isLast, path):
    eval_returns = []
    eval_ep_lens = []
    o, ep_ret, ep_len, = env.reset(), 0, 0
    n = 0
    lim = 100 if isLast else 10
    while n < lim:
        # a, _, _ = pi.step(torch.as_tensor(o, dtype=torch.float32), training=True)
        a, _, _ = pi.step(torch.as_tensor(o, dtype=torch.float32), training=False)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            eval_returns.append(ep_ret)
            eval_ep_lens.append(ep_len)
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            n += 1
    
    if isLast:
        dir_path = os.path.dirname(path)
        eval_file_path = os.path.join(dir_path, "last_eval.txt")
        with open(eval_file_path, 'w') as f:
            for ret in eval_returns:
                f.write(str(ret) + '\n')


    return eval_returns, eval_ep_lens
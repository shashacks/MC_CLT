# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms
from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
from spinup.algos.tf1.sac.sac import sac as sac_tf1
from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1

from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.ppo_ensemble.ppo_ensemble import ppo_ensemble as ppo_ensemble_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch
from spinup.algos.pytorch.dppo.dppo import dppo as dppo_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.trpo_ensemble.trpo_ensemble import trpo_ensemble as trpo_ensemble_pytorch
from spinup.algos.pytorch.dtrpo.dtrpo import dtrpo as dtrpo_pytorch

from spinup.algos.pytorch.rn_ppo.rn_ppo import rn_ppo as rn_ppo_pytorch
from spinup.algos.pytorch.rn_trpo.rn_trpo import rn_trpo as rn_trpo_pytorch


# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
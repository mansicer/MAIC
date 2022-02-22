from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .maic_learner import MAICLearner
from .maic_qplex_learner import MAICQPLEXLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY['maic_learner'] = MAICLearner
REGISTRY['maic_qplex_learner'] = MAICQPLEXLearner

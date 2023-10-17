import sys
sys.path.append("/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-vaep-model/src")
sys.path.append("/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-vaep-model/src/features")
sys.path.append("/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-score-model")

from models.predict_model import predict_model
from models.predict_player_stats import predict_player_stats

predict_model()
predict_player_stats()
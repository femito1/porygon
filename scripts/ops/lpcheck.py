import sys
from porygon.v2.evaluate import load_player
p = load_player(sys.argv[1] + ",temperature=0.5", battle_format="gen9randombattle", start_listening=False)
print("load_player OK:", type(p).__name__, getattr(p, "temperature", None))

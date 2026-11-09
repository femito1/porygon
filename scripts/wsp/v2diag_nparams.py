"""Print the parameter count of a v2 checkpoint (as the v2 trainer reports it: sum of net.parameters() sizes)."""
import sys

import porygon.v2.model as M

print(sum(p.numel() for p in M.load(sys.argv[1]).parameters()))

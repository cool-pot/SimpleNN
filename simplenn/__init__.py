"""SimpleNN -- Simple Neural Network implementation in the idea of computation graph using pure python3

This repo is for academic usage only, no guarantee, although will try, for performance and accuracy,

License: Creative Commons Attribution 4.0 International License

Acknowledge: This computation graph framework was designed by
Philipp Meerkamp, Pierre Garapon, and David Rosenberg. This is a python3 implementation
by Yi Zhou after taking David Rosenberg's DS-GA 1003 Machine Learning Course at NYU Data
Science Center and for the sake of CS-GY 6643 Computer Vision Project.

Author: Yi Zhou
"""
import simplenn.nodes as nodes
import simplenn.graph as graph
import simplenn.models as models
import simplenn.utility as utility

try:
    get_ipython
    print("🤖 IPython detected")
    print("😎 SimpleNN successfully imported. Have fun.")

except Exception:
    print("🤖 Shell detected")
    print("😎 SimpleNN successfully imported. Have fun.")


# module level doc-string
__doc__ = """
SimpleNN -- Simple Neural Network implementation in the idea of computation graph using pure python3
"""
"""
Root convenience entrypoint.

Allows:
  python train_distributed.py --mode pbt ...
"""

from tools.train_distributed import main


if __name__ == "__main__":
    main()


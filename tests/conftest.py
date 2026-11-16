import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['WANDB_DISABLED']='true'
sys.path.insert(0,str(ROOT/'shared_dual'))
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(4)

import faulthandler
faulthandler.enable()  # stampa traceback C/Python in caso di crash

import sys
import os

# Aggiunge la root del package 'experiments' al path così gli import assoluti
# funzionano sia con `python train.py` che con `python -m experiments.train.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging import getLogger
from hopwise.config import Config
from hopwise.data import create_dataset, data_preparation
from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM
from hopwise.trainer import Trainer
from hopwise.utils import init_seed, init_logger
from hopwise.quick_start.quick_start import run_hopwise, run
from transformers.trainer_utils import get_last_checkpoint
from utils.dataloader import load_data
import neo4j
import glob
import shutil
from datetime import datetime


def find_pretrained_hf_checkpoint(saved_dir: str) -> str:
    """
    Trova l'ultima directory checkpoint HuggingFace salvata dal pretrain KGGLM.
    Hopwise salva in: saved/huggingface-distilgpt2-KGGLM-autism-pretrained-{N}.pth/checkpoint-{step}/
    Restituisce il path al checkpoint più recente per epoch (N più alto).
    """
    pattern = os.path.join(saved_dir, "huggingface-*-pretrained-*.pth")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Nessun checkpoint pretrain trovato in {saved_dir}")
    # prende l'epoch più alta (ultimo nel sort lessicografico)
    last_epoch_dir = candidates[-1]
    ckpt = get_last_checkpoint(last_epoch_dir)
    if ckpt is None:
        raise FileNotFoundError(f"Nessun checkpoint HuggingFace trovato in {last_epoch_dir}")
    return ckpt

if __name__ == '__main__':

    if len(sys.argv) > 1 and "data" in sys.argv:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
            with driver.session() as session:
                load_data(session, "../data/autism")
        print("Data loaded successfully.")
        sys.exit(0)

    if len(sys.argv) > 1 and "test" in sys.argv:
        run_hopwise(
            model='KGGLM',
            dataset='autism',
            config_file_list=['kgglm.yaml'],
            saved=False,
            config_dict={'train_stage': 'pretrain', 'pretrain_epochs': 1}
        )
        sys.exit(0)

    if len(sys.argv) > 1 and "run" in sys.argv:
        config_dict = {
            'train_stage': 'pretrain',
            'pretrain_epochs': 20,
        }

        run_hopwise(
            model='KGGLM',
            dataset='autism',
            config_file_list=['kgglm.yaml'],
            saved=True,
            config_dict=config_dict
        )

        saved_dir = os.path.join(os.getcwd(), "saved")
        pre_model_path = find_pretrained_hf_checkpoint(saved_dir)

        config_dict = {
            'train_stage': 'finetune',
            'pre_model_path': pre_model_path,
        }

        run_hopwise(
            model='KGGLM', # 'PEARLM' 'KGGLM'
            dataset='autism', # 'autism' 'ml-100k'
            config_file_list=['kgglm.yaml'], # 'hopwise.yaml'
            config_dict=config_dict,
            saved=True,
            checkpoint=None
        )

        src_dir = os.path.join(os.getcwd(), "saved")
        dst_dir = os.path.join(
            os.getcwd(),
            "trained",
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

        if os.path.isdir(src_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for name in os.listdir(src_dir):
                src_path = os.path.join(src_dir, name)
                dst_path = os.path.join(dst_dir, name)
                shutil.move(src_path, dst_path)

import faulthandler
faulthandler.enable()  # stampa traceback C/Python in caso di crash

from logging import getLogger
from hopwise.config import Config
from hopwise.data import create_dataset, data_preparation
from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM
from hopwise.trainer import Trainer
from hopwise.utils import init_seed, init_logger
from hopwise.quick_start.quick_start import run_hopwise, run
import os
import shutil
from datetime import datetime

if __name__ == '__main__':
    run_hopwise(
        model='PEARLM',
        dataset='autism', # 'autism' 'ml-100k'
        run='train',
        config_file_list=['experiments.yaml'], # 'hopwise.yaml'
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

    """
    config = Config(model='PEARLM', dataset='autism', config_file_list=['hopwise.yaml'])

    # init random seed
    init_seed(seed=123, reproducibility=True)

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # model loading and initialization
    model = PEARLM(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)"""

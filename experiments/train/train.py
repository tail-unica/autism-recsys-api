import faulthandler
faulthandler.enable()  # stampa traceback C/Python in caso di crash

from logging import getLogger
from hopwise.config import Config
from hopwise.data import create_dataset, data_preparation
from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM
from hopwise.trainer import Trainer
from hopwise.utils import init_seed, init_logger
from hopwise.quick_start.quick_start import run_hopwise, run

if __name__ == '__main__':
    run_hopwise(
        model='PEARLM',
        dataset='autism', # 'autism' 'ml-100k'
        run='train',
        config_file_list=['hopwise.yaml'], # 'hopwise.yaml'
        saved=False,
        checkpoint=None
    )

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

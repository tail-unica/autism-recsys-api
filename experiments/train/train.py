from logging import getLogger
from hopwise.config import Config
from hopwise.data import create_dataset, data_preparation
from hopwise.model.general_recommender import BPR
from hopwise.trainer import Trainer
from hopwise.utils import init_seed, init_logger

if __name__ == '__main__':
    config = Config(model='BPR', dataset='autism', config_file_list=['hopwise.yaml'])

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
    model = BPR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)

from src.tools.startup import params, logger
from src.models import general
from src.data import disk
import sys
import os


def execute_pipeline(settings, model):
    """
    Execute main pipeline
    """
    logger.info(f'Starting execution: {settings["exec_name"]}')
    settings['hyperparameter_search']['tune_params']['name'] =\
        settings['exec_name'] + '_' + model

    logger.info('Running grid search')
    best_params = general.grid_search(settings['hyperparameter_search'], model)

    logger.info('Storing results')
    objects = {
        'settings': settings,
        'best_params': best_params
    }
    disk.store_output(objects, 
        settings['hyperparameter_search']['output_path'], settings['exec_name'])

    logger.info('Execution finished')


if __name__ == '__main__':
    available_args = [
        'resnet.LSTM', 'inception_resnet.LSTM',
        'resnet.TCN', 'inception_resnet.TCN']
    arg = sys.argv[1]
    if arg not in available_args:
        raise Exception(f'Incorrect argument. Options are: {available_args}')
    execute_pipeline(params, arg)
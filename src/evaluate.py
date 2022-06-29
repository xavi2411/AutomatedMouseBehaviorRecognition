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

    logger.info('Training and testing model')
    figures_output = os.path.join(
        settings['evaluation']['output_path'], settings['exec_name'])
    os.mkdir(figures_output)
    general.train_test_model(settings['evaluation'], model, figures_output)

    logger.info('Storing results')
    objects = {
        'settings': settings['evaluation'],
    }
    disk.store_output(objects, 
        settings['evaluation']['output_path'], settings['exec_name'], create=False)

    logger.info('Execution finished')

if __name__ == '__main__':
    available_args = [
        'resnet.LSTM', 'inception_resnet.LSTM',
        'resnet.TCN', 'inception_resnet.TCN']
    arg = sys.argv[1]
    if arg not in available_args:
        raise Exception(f'Incorrect argument. Options are: {available_args}')
    execute_pipeline(params, arg)
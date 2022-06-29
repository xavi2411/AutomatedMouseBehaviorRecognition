from src.tools.startup import params, logger
from src.data import processing

def execute_pipeline(settings):
    """
    Execute main pipeline
    """
    logger.info(f'Starting execution: {settings["exec_name"]}')

    logger.info('Generating dataset')
    processing.generate_dataset(settings['dataset'])

    logger.info('Execution finished')

if __name__ == '__main__':
    execute_pipeline(params)
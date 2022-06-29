import os
import pandas as pd
import numpy as np
from ray import tune
from tensorflow import keras
from src.data import disk, processing
from src.models import modelling

METRIC_NAMES = [
    'TruePositives',
    'FalsePositives',
    'TrueNegatives',
    'FalseNegatives',
    'Accuracy',
    'Precision',
    'Recall',
    'PRC',
]

BEHAVIOR_NAMES = [
    'Grooming',
    'Mid Rearing',
    'Wall Rearing',
]


def train_test_model(settings, model):
    """
    Train and test the selected model
    """


def grid_search(settings, model):
    """
    Perform a grid search of the parameters given.
    """
    config = {}
    for key, value in settings[model].items():
        if isinstance(value, list):
            config[key] = tune.grid_search(value)
        else:
            config[key] = value

    analysis = tune.run(
        tune.with_parameters(tune_train, settings=settings['parameters']),
        config=config,
        metric=settings['tune_params']['metric'],
        mode=settings['tune_params']['mode'],
        resources_per_trial=settings['tune_params']['resources'],
        resume=settings['tune_params']['resume'],
        name=settings['tune_params']['name'],
        raise_on_failed_trial=settings['tune_params']['raise_on_fail'],
        verbose=settings['tune_params']['verbose'],
    )
    return analysis.get_best_config(
        metric=settings['tune_params']['metric'], 
        mode=settings['tune_params']['mode'])


def tune_train(config, settings):
    keras.backend.clear_session()
    tune.utils.wait_for_gpu(target_util=0.2)
    # Load data
    dataset = disk.load_dataset(
        os.path.join(settings['abs_path'], settings['input_path'], config['backbone']),
        settings['split'], ['train'])

    # Split data into sequences
    X, y = processing.generate_sequences(dataset['train'], config['sequence_length'])
    sequences_per_video = X.shape[0] // dataset['train']['num_videos']

    final_metrics = pd.DataFrame(columns=METRIC_NAMES)
    # Loop over training videos (Leave one out strategy)
    for v_idx in range(dataset['train']['num_videos']):
        # Split into train and validation
        X_train = np.concatenate(
            (X[:v_idx*sequences_per_video], X[(v_idx+1)*sequences_per_video:]), axis=0)
        y_train = np.concatenate(
            (y[:v_idx*sequences_per_video], y[(v_idx+1)*sequences_per_video:]), axis=0)
        X_val = X[v_idx*sequences_per_video:(v_idx+1)*sequences_per_video]
        y_val = y[v_idx*sequences_per_video:(v_idx+1)*sequences_per_video]

        b_metrics = pd.DataFrame(columns=METRIC_NAMES)
        # Loop over behaviors
        for b_idx in range(y.shape[2]):
            y_behavior_train = y_train[:,:,b_idx:b_idx+1]
            y_behavior_val = y_val[:,:,b_idx:b_idx+1]
            # Compute sample weights
            sample_weights = modelling.compute_sample_weights(y_behavior_train)
            # Create model
            model = modelling.make_model(
                X_train.shape[1:], config['layers'], config['dropout_rate'],
                config['num_layers'], config['num_units'],
                kernel_size=config.get('kernel_size'), norm=config.get('norm'),
                loss=config['loss'], optimizer=config['optimizer'],
                learning_rate=config['learning_rate'],
            )
            # Train model
            model.fit(
                X_train, y_behavior_train,
                validation_data=(X_val, y_behavior_val),
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                verbose=False,
                shuffle=True,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor=config['es_monitor'],
                        mode=config['es_mode'],
                        patience=config['es_patience'],
                        restore_best_weights=True,
                        verbose=False,
                    ),
                ],
                sample_weight=sample_weights,
            )
            # Evaluate model
            loss, *metrics = model.evaluate(X_val, y_behavior_val, verbose=False)
            # Store the metrics
            b_metrics = pd.concat([
                b_metrics,
                pd.DataFrame([metrics], columns=METRIC_NAMES)
            ])
            del model
        # Aggregate and storebehavior metrics
        final_metrics = pd.concat([
            final_metrics,
            pd.DataFrame([b_metrics.mean(axis=0)], columns=METRIC_NAMES)
        ])
    # Aggregate validation metrics
    result = final_metrics.mean(axis=0)
    # Report the metrics to Ray tune
    tune.report(accuracy=result['Accuracy'])
    keras.backend.clear_session()
import os
import pandas as pd
import numpy as np
from ray import tune
from tensorflow import keras
from src.data import disk, processing
from src.models import modelling
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)
import seaborn as sns


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


def train_test_model(settings, model, figures_output):
    """
    Train and test the selected model
    """
    config = settings[model]
    # Load data
    dataset = disk.load_dataset(
        os.path.join(settings['input_path'], config['backbone']),
        settings['split'], ['train', 'test'])

    # Split data into sequences
    X_train, y_train = processing.generate_sequences(
        dataset['train'], config['sequence_length'])
    X_test, y_test = processing.generate_sequences(
        dataset['test'], config['sequence_length'])

    final_metrics = pd.DataFrame(columns=['Behavior'] + METRIC_NAMES)
    for b_idx in range(y_train.shape[2]):
        print('Running experiment for {}'. format(BEHAVIOR_NAMES[b_idx]))
        # Preserve the behavior labels
        y_behavior_train = y_train[:,:,b_idx:b_idx+1]
        y_behavior_test = y_test[:,:,b_idx:b_idx+1]

        # Compute sample weights
        sample_weights = modelling.compute_sample_weights(y_behavior_train)
        # Create model
        model = modelling.make_model(
            X_train.shape[1:], config['layers'], config['dropout_rate'],
            config['num_layers'], config['num_units'],
            loss=config['loss'], optimizer=config['optimizer'],
            learning_rate=config['learning_rate'],
        )
        # Train model
        history = model.fit(
            X_train, y_behavior_train,
            validation_data=(X_test, y_behavior_test),
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
        loss, *metrics = model.evaluate(X_test, y_behavior_test, verbose=False)
        # Print and plot the results
        print('Results:')
        print('\tLoss: {}'.format(loss))
        for idx, metric in enumerate(metrics):
            print('\t{}: {}'.format(METRIC_NAMES[idx], metric))

        output_path = os.path.join(figures_output, BEHAVIOR_NAMES[b_idx].lower())
        os.mkdir(output_path)
        # Plot: loss and acc
        plot_metrics(history, output_path)
        plt.close()
        # Plot Confusion matric and Precision-Recall curve
        train_predictions = model.predict(X_train, batch_size=config['batch_size'])
        test_predictions = model.predict(X_test, batch_size=config['batch_size'])
        plot_cm(y_behavior_test.flatten(), test_predictions.flatten(), output_path)
        plt.close()
        plot_prc("Train Baseline", y_behavior_train.flatten(), train_predictions.flatten(), output_path)
        plot_prc("Test Baseline", y_behavior_test.flatten(), test_predictions.flatten(), output_path, linestyle='--')
        plt.legend(loc='lower right')
        plt.close()
        # Store the metrics
        final_metrics = pd.concat([
            final_metrics,
            pd.DataFrame([[BEHAVIOR_NAMES[b_idx]] + metrics], columns=['Behavior'] + METRIC_NAMES)
        ])
        del model
    return final_metrics

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


def plot_cm(labels, predictions, output_path=None, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if output_path is not None:
        plt.savefig(os.path.join(output_path, 'cm.png'))


def plot_prc(name, labels, predictions, output_path=None, **kwargs):
    precision, recall, _ = precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    if output_path is not None:
        plt.savefig(os.path.join(output_path, 'prc.png'))


def plot_metrics(history, output_path=None):
    metrics = ['loss', 'accuracy']
    f, axs = plt.subplots(1, 2)
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        axs[n].plot(history.epoch, history.history[metric], label='Train')
        axs[n].plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Test')
        axs[n].set_xlabel('Epoch')
        axs[n].set_ylabel(name)
        if metric == 'loss':
            axs[n].set_ylim([0, axs[n].set_ylim()[1]])
        else:
            axs[n].set_ylim([0,1])

        axs[n].legend()
    if output_path is not None:
        plt.savefig(os.path.join(output_path, 'metrics.png'))
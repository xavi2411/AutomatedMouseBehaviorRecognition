import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from tensorflow import keras
from tcn import TCN
from ray import tune
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

def load_dataset(path, backbone, sets=['train', 'test']):
    '''
    Load the dataset
    '''
    if backbone not in ['resnet', 'inception_resnet']:
        raise Exception('Invalid backbone')
    dataset = {}
    for set in sets:
        feature_files = [f for f in sorted(os.listdir(os.path.join(path, backbone, set, 'features')))]
        label_files = [f for f in sorted(os.listdir(os.path.join(path, backbone, set, 'labels')))]
        features = []
        labels = []
        for f, l in zip(feature_files, label_files):
            features.append(np.load(os.path.join(path, backbone, set, 'features', f)))
            labels.append(pd.read_csv(os.path.join(path, backbone, set, 'labels', l)).values)
        dataset[set] = {
            'features': np.concatenate(features, axis=0),
            'labels': np.concatenate(labels, axis=0),
            'num_videos': len(feature_files),
        }
    return dataset


def generate_sequences(data, seq_length):
    '''
    Generate the sequences by splitting the data
    '''
    X = np.array([data['features'][i:i+seq_length] for i in range(0, data['features'].shape[0], seq_length)])
    y = np.array([data['labels'][i:i+seq_length] for i in range(0, data['labels'].shape[0], seq_length)])
    return X, y


def compute_sample_weights(y_train):
    neg, pos = np.bincount(y_train.flatten())

    total = pos + neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    sample_weights = []
    for row in y_train:
        row_weights = []
        for column in row:
            col_weights = []
            for value in column:
                if value == 0:
                    col_weights.append(weight_for_0)
                else:
                    col_weights.append(weight_for_1)
            row_weights.append(np.array(col_weights))
        sample_weights.append(np.array(row_weights))

    return np.array(sample_weights)


def make_model(input_shape, layers, dropout_rate=0.5, num_layers=1, num_units=128, kernel_size=3, norm='batch', loss='binary_crossentropy', optimizer='adam', learning_rate=0.001):
    """
    Create a model based on the specified config
    """
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='prc', curve='PR')
    ]

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape, name='Input'))
    model.add(keras.layers.Dropout(dropout_rate))
    if layers == 'lstm':
        model.add(keras.layers.LSTM(num_units, return_sequences=True))
        for i in range(num_layers - 1):
            num_units = num_units // 2
            model.add(keras.layers.LSTM(num_units, return_sequences=True))
    elif layers == 'tcn':
        batch, layer, weight = False, False, False
        if norm == 'batch':
            batch = True
        if norm == 'layer':
            layer = True
        if norm == 'weight':
            weight = True
        model.add(TCN(num_units, kernel_size=kernel_size, use_batch_norm=batch, use_layer_norm=layer, use_weight_norm=weight, return_sequences=True))
        for i in range(num_layers - 1):
            num_units = num_units // 2
            model.add(TCN(num_units, kernel_size=kernel_size, use_batch_norm=batch, use_layer_norm=layer, use_weight_norm=weight, return_sequences=True))
    else:
        raise ValueError('Layer not supported.')   
 
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=METRICS
    )
    return model


def tune_train(config):
    keras.backend.clear_session()
    tune.utils.wait_for_gpu(target_util=0.2)
    # Load data
    dataset = load_dataset(os.path.join(config['abs_path'], 'data/processed/Dataset'), config['backbone'], ['train'])

    # Split data into sequences
    X, y = generate_sequences(dataset['train'], config['sequence_length'])
    sequences_per_video = X.shape[0] // dataset['train']['num_videos']

    final_metrics = pd.DataFrame(columns=METRIC_NAMES)
    # Loop over training videos (Leave one out strategy)
    for v_idx in range(dataset['train']['num_videos']):
        # Split into train and validation
        X_train = np.concatenate((X[:v_idx*sequences_per_video], X[(v_idx+1)*sequences_per_video:]), axis=0)
        y_train = np.concatenate((y[:v_idx*sequences_per_video], y[(v_idx+1)*sequences_per_video:]), axis=0)
        X_val = X[v_idx*sequences_per_video:(v_idx+1)*sequences_per_video]
        y_val = y[v_idx*sequences_per_video:(v_idx+1)*sequences_per_video]

        b_metrics = pd.DataFrame(columns=METRIC_NAMES)
        # Loop over behaviors
        for b_idx in range(y.shape[2]):
            y_behavior_train = y_train[:,:,b_idx:b_idx+1]
            y_behavior_val = y_val[:,:,b_idx:b_idx+1]
            # Compute sample weights
            sample_weights = compute_sample_weights(y_behavior_train)
            # Create model
            model = make_model(
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

def grid_search(config):
    '''
    Perform a grid search of the parameters given.
    '''
    for key, value in config.items():
        if isinstance(value, list):
            config[key] = tune.grid_search(value)

    analysis = tune.run(
        tune_train,
        config=config,
        metric='accuracy',
        mode='max',
        resources_per_trial={'cpu': 12, 'gpu': 1},
        # resume=True,
        # name='tune_train_2022-05-18_19-16-40',
        raise_on_failed_trial=False,
        verbose=1,
    )
    return analysis.get_best_config(metric='accuracy', mode='max')


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_metrics(history):
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

def train_test_model(config):
    # Load data
    dataset = load_dataset(os.path.join(config['abs_path'], 'data/processed/Dataset'), config['backbone'], ['train', 'test'])

    # Split data into sequences
    X_train, y_train = generate_sequences(dataset['train'], config['sequence_length'])
    X_test, y_test = generate_sequences(dataset['test'], config['sequence_length'])

    final_metrics = pd.DataFrame(columns=['Behavior'] + METRIC_NAMES)
    for b_idx in range(y_train.shape[2]):
        print('Running experiment for {}'. format(BEHAVIOR_NAMES[b_idx]))
        # Preserve the behavior labels
        y_behavior_train = y_train[:,:,b_idx:b_idx+1]
        y_behavior_test = y_test[:,:,b_idx:b_idx+1]

        # Compute sample weights
        sample_weights = compute_sample_weights(y_behavior_train)
        # Create model
        model = make_model(
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
        # Plot: loss and acc
        plot_metrics(history)
        plt.show()
        # Plot Confusion matric and Precision-Recall curve
        train_predictions = model.predict(X_train, batch_size=config['batch_size'])
        test_predictions = model.predict(X_test, batch_size=config['batch_size'])
        plot_cm(y_behavior_test.flatten(), test_predictions.flatten())
        plt.show()
        plot_prc("Train Baseline", y_behavior_train.flatten(), train_predictions.flatten())
        plot_prc("Test Baseline", y_behavior_test.flatten(), test_predictions.flatten(), linestyle='--')
        plt.legend(loc='lower right')
        plt.show()
        # Store the metrics
        final_metrics = pd.concat([
            final_metrics,
            pd.DataFrame([[BEHAVIOR_NAMES[b_idx]] + metrics], columns=['Behavior'] + METRIC_NAMES)
        ])
        del model
    return final_metrics
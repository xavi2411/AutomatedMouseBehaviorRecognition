import os
import sys
import cv2
from tqdm import tqdm
import numpy as np
from src.data import disk
from src.models.feature_extractor import FeatureExtractor
from src.models import modelling

def generate_dataset(settings, crop_expansion=80):
    """
    Load, generate and save dataset
    """
    # Get the raw input paths
    videos_src_folder = os.path.join(
        settings['input']['path'], settings['input']['data']['videos'])
    csvs_src_folder = os.path.join(
        settings['input']['path'], settings['input']['data']['csvs'])
    excels_src_folder = os.path.join(
        settings['input']['path'], settings['input']['data']['excels'])

    # Get the input filenames
    excels = sorted(
        [f for f in os.listdir(excels_src_folder) if f.endswith('.xlsx')],
        key=lambda x: x.split('.')[0][-4:])
    csvs = sorted(
        [f for f in os.listdir(csvs_src_folder) if f .endswith('csv')], 
        key=lambda x: x.split('.')[0][-4:])
    videos = sorted(
        [f for f in os.listdir(videos_src_folder) if f.endswith('.mp4')])

    # Get and create output folders
    output_folder = os.path.join(settings['output']['path'],
        settings['feature_extractor'])
    output_features_folder = os.path.join(output_folder, 'features')
    output_labels_folder = os.path.join(output_folder, 'labels')
    try:
        os.mkdir(output_folder)
        os.mkdir(output_features_folder)
        os.mkdir(output_labels_folder)
    except:
        print('Couldn\'t create output folders')
        sys.exit()

    # Create the FeatureExtractor
    feature_extractor = FeatureExtractor(settings['feature_extractor'])

    for excel, csv, video in tqdm(zip(excels, csvs, videos)):
        # Get and save labels from excel file
        df_excel = disk.load_excel(os.path.join(excels_src_folder, excel),
            header=0, usecols='E,F,G')
        df_excel = preprocess_excel(df_excel)
        disk.store_labels(df_excel, 
            os.path.join(output_labels_folder, excel.split('.')[0] + '.csv'), index=False)

        # Split video frames crop by the midbody position and run the Feature Extractor
        df = disk.load_csv(os.path.join(csvs_src_folder, csv), header=[0,1,2])
        df = preprocess_csv(df)
        
        # Get the midbody position
        midbody = extract_midbody_position(df)

        # Get video frames
        video_features = preprocess_video(os.path.join(
            videos_src_folder, video), midbody, crop_expansion, feature_extractor)
        disk.store_features(os.path.join(
            output_features_folder, csv.split('.')[0] + '.npy'), video_features)


def preprocess_excel(df):
    """
    Preprocess the excel file
    """
    df.drop(index=[0,1], inplace=True)
    df.fillna(0, inplace=True)
    df.replace(["x", "X"], 1, inplace=True)
    df.columns = [col.lower().replace(' ','_') for col in df.columns]
    df = df.astype(int)
    df.reset_index(inplace=True, drop=True)
    return df


def preprocess_csv(df):
    """
    Preprocess the csv file
    """
    df.columns = [('%s%s%s' % 
        ('%s' % a.lower() if not a.startswith("DLC") else '',
        ' %s' % b.lower() if not b.startswith("Unnamed") else '',
        ' %s' % c.lower() if not c.startswith("Unnamed") else '')
        ).strip().replace(' ', '_') 
        for a,b,c in df.columns
    ]
    df.set_index(df.columns[0], inplace=True)
    return df


def extract_midbody_position(df):
    """
    Extract the mouse midbody by using the df information
    """
    midbody = np.concatenate((
        df['midbody_y'].values[:, np.newaxis], 
        df['midbody_x'].values[:, np.newaxis]), axis=1)
    midbody = midbody.astype(int)
    return midbody


def preprocess_video(file, midbody, expansion, feature_extractor):
    """
    Preprocess each video frame
    """
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    count = 0
    features = []
    while success:
        # Build frame name
        frame_name = 'frame'
        for i in range(4-len(str(count))):
            frame_name += '0'
        frame_name += str(count) + '.jpg'

        # Crop image based on mouse midbody (make boxes having the mouse in the middle, should be square)
        top = max(0, midbody[count][0] - expansion) - max(0, midbody[count][0] + expansion - image.shape[0])
        bottom = min(image.shape[0], midbody[count][0] + expansion) + max(0, expansion - midbody[count][0])
        left = max(0, midbody[count][1] - expansion) - max(0, midbody[count][1] + expansion - image.shape[1])
        right = min(image.shape[1], midbody[count][1] + expansion) + max(0, expansion - midbody[count][1])
        frame = image[top:bottom, left:right]

        # Run the feature extractor
        frame_emb = modelling.generate_frame_embedding(
            frame, feature_extractor)
        features.append(frame_emb)

        success, image = vidcap.read()
        count += 1

    return features


def generate_sequences(data, seq_length):
    '''
    Generate the sequences by splitting the data
    '''
    X = np.array([data['features'][i:i+seq_length] 
        for i in range(0, data['features'].shape[0], seq_length)])
    y = np.array([data['labels'][i:i+seq_length] 
        for i in range(0, data['labels'].shape[0], seq_length)])
    return X, y
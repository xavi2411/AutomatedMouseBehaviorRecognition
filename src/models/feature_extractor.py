from tabnanny import verbose
import tensorflow as tf

class FeatureExtractor(tf.keras.Model):
    def __init__(self, model_name):
        """
            Initialize Feature extractor with a pretrained CNN model

            Args:
                model_name: name of the pretrained CNN model ["resnet", "inception_resnet"]
        """
        super(FeatureExtractor, self).__init__()
        if model_name == "resnet":
            from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
            self.model = ResNet50V2(include_top=False, weights='imagenet', pooling='avg')
            self.model_input_size = (224, 224)
        elif model_name == "inception_resnet":
            from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
            self.model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
            self.model_input_size = (299, 299)
        else:
            raise NameError('Invalid pretrained model name - must be one of ["resnet", "inception_resnet"]')
        
        self.preprocess_input = preprocess_input
        self.model.trainable = False

    def call(self, inputs):
        """
            Call the pretrained CNN model to predict the features for a given input image

            Args:
                inputs: input image tensor
        """
        # Resize inputs to the expected input size
        inputs = inputs*255
        inputs = tf.image.resize(inputs, self.model_input_size)
        inputs = inputs[tf.newaxis, :]
        preprocessed_input = self.preprocess_input(inputs)
        return self.model.predict(preprocessed_input, verbose=False).ravel()
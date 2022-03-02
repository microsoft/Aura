"""VGGish model for Keras. A VGG-like model for audio classification
# Reference
- [CNN Architectures for Large-Scale Audio Classification](ICASSP 2017)
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D


class VGGish(Model):
    """
    VGGIsh implementation in keras -- generate as 128-feature embedding from audio files
    """

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')
        self.conv2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')
        self.conv3_1 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')
        self.conv3_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')
        self.conv4_1 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')
        self.conv4_2 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')

        self.maxpool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')
        self.maxpool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')
        self.max_pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')
        self.max_pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')

    def estimate(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.max_pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.max_pool4(x)

        return GlobalAveragePooling2D()(x), x

    def call(self, x):
        return self.estimate(x)


if __name__ == '__main__':
    WEIGHT_PATH = 'C:\\Users\\t-xgitiaux\\OneDrive - Microsoft\\saved_models\\vggish\\'
    model = VGGish()
    model.load_weights(f'{WEIGHT_PATH}\\vggish_model_resaved.ckpt')

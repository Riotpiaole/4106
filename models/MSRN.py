import numpy as np
import tensorflow as tf

from keras.layers import (
    Input,
    concatenate,
    Add
)
from keras.layers.convolutional import Conv2D


def msrn_block(inputs, channel=64):
    layer_3_1 = Conv2D(
        channel,
        (3, 3),
        padding='same',
        use_bias=True,
        activation='relu'
    )(inputs)

    layer_5_1 = Conv2D(
        channel,
        (5, 5),
        padding='same',
        use_bias=True,
        activation='relu'
    )(inputs)

    input_2 = concatenate([layer_3_1, layer_5_1], axis=3)

    layer_3_2 = Conv2D(
        channel * 2,
        (3, 3),
        padding='same',
        use_bias=True,
        activation='relu'
    )(input_2)

    layer_5_2 = Conv2D(
        channel * 2,
        (5, 5),
        padding='same',
        use_bias=True,
        activation='relu'
    )(input_2)

    second_con = concatenate([layer_3_2, layer_5_2], axis=3)

    confusion = Conv2D(
        channel * 4,
        (1, 1),
        padding='same',
        use_bias=False
    )(second_con)
    output = Conv2D(
        channel,
        (1, 1),
        padding='same',
        use_bias=False
    )(confusion)
    return output


def pixel_shuffling(scale, input):
    pass


def MSRN(input, input_dim=[32, 32, 1], channel=64, scale=2):
    x = Input(shape=[32, 32, 1])

    x = Conv2D(64, (3, 3), padding="same")(x)

    residual1 = msrn_block(x)

    residual2 = msrn_block(residual1)

    residual3 = msrn_block(residual2)

    residual4 = msrn_block(residual3)

    residual5 = msrn_block(residual4)

    residual6 = msrn_block(residual5)

    residual7 = msrn_block(residual6)

    residual8 = msrn_block(residual7)

    output = concatenate(
        [
            x,
            residual1,
            residual2,
            residual3,
            residual4,
            residual5,
            residual6,
            residual7,
            residual8,
        ],
        3
    )

    bottle = Conv2D(channel, (1, 1), padding='same', use_bias=True)(output)
    conv = Conv2D(channel * scale * scale, (3, 3),
                  padding='same', use_bias=True)(bottle)
    # TODO add pixel Shuffling
    convt = pixel_shuffling(scale,conv)

    out = Conv2D(1,(3,3),padding='same', use_bias=True)



if __name__ == "__main__":
    x = Input(shape=[32, 32, 1])
    x = Conv2D(
        64,
        (3, 3),
        padding='same'
    )(x)
    a = msrn_block(x)

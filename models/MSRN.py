import numpy as np
import tensorflow as tf

from keras.layers import (
    Input,
    concatenate,
    Add
)
from keras.layers.convolutional import Conv2D

def msrn_block(inputs,channel=64):
    layer_3_1 = Conv2D(
        channel,
        (3,3),
        padding='same',
        use_bias=True,
        activation='relu'
    )(inputs)

    layer_5_1 = Conv2D(
        channel,
        (5,5),
        padding='same',
        use_bias=True,
        activation='relu'
    )(inputs)

    input_2 = concatenate([layer_3_1 , layer_5_1],axis=1)

    layer_3_2 = Conv2D(
        channel*2,
        (3,3),
        padding='same',
        use_bias=True,
        activation='relu'
    )( input_2 )

    layer_5_2 = Conv2D(
        channel*2,
        (5,5),
        padding='same',
        use_bias=True,
        activation='relu'
    )(input_2)

    second_con = concatenate([layer_3_2 , layer_5_2 ],axis=0)

    confusion = Conv2D(
        channel*4,
        (1,1),
        padding='same',
        use_bias=False
    )(second_con)
    # output = Conv2D(
    #     channel*2,
    #     (1,1),
    #     padding='same',
    #     use_bias=False
    # )(confusion)
    return confusion



if __name__ == "__main__":
    x = Input( shape=[32,32,3] )
    x = Conv2D(
        64,
        (3,3),
        padding='same'
    )(x)
    a = msrn_block(x)

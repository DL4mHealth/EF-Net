import tensorflow as tf
from keras.models import Model


from tensorflow.keras.layers import Flatten, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, Lambda, Dense
from tensorflow.keras import Input



from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Normalization



#EF-Net
def eeg_fnirs_cnn_v2():
    tf.keras.backend.clear_session()

    inputE = Input(shape=(500, 30,1))
    inputF = Input(shape=(25, 72,1))

    e = Conv2D(filters=32, kernel_size=(7,1), activation='relu')(inputE)   
    e = Conv2D(filters=32, kernel_size=(7,1), activation='relu')(e)
    e = Conv2D(filters=32, kernel_size=(7,1), activation='relu')(e)
    e= MaxPooling2D(pool_size=(7,1))(e)   
    e= Dropout(0.5)(e)
    e= BatchNormalization()(e)

    e =Conv2D(filters=64, kernel_size=(4,4),activation='relu')(e)             
    e =Conv2D(filters=64, kernel_size=(4,4),activation='relu')(e)
    e =Conv2D(filters=64, kernel_size=(4,4),activation='relu')(e)
    e= MaxPooling2D(pool_size=(4,4))(e)
    e= Dropout(0.5)(e)
    e= BatchNormalization()(e)
    e= Flatten()(e)

    e=Dense(256,activation="relu")(e)
    e= Dropout(0.5)(e)
    e=Dense(128,activation="relu")(e)
    e = Model(inputs=inputE, outputs=e)


    f=Conv2D(filters=32, kernel_size=(4,1))(inputF)    
    f=Conv2D(filters=32, kernel_size=(4,1))(f)
    f= MaxPooling2D(pool_size=(4,1))(f)  
    f= Dropout(0.5)(f)
    f= BatchNormalization()(f)
    f=Conv2D(filters=64, kernel_size=(2,2))(f)
    f=Conv2D(filters=64, kernel_size=(2,2))(f)
    f= MaxPooling2D(pool_size=(2,2))(f)
    f= Dropout(0.5)(f)
    f= BatchNormalization()(f)
    f= Flatten()(f)
    f=Dense(128,activation="relu")(f)
    f = Model(inputs=inputF, outputs=f)

    combined = concatenate([e.output, f.output])
    #MLP
    z = Dense(256, activation="relu")(combined)
    z= Dropout(0.5)(z)
    z= Lambda(lambda x: tf.math.l2_normalize(x,axis=1, epsilon=5e-4))(z)    #L2 norm
    z = Dense(64, activation="relu")(z)
    z = Dense(1, activation="sigmoid")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[e.input, f.input], outputs=z)
    return model


#Baseline
def eeg_fnirs_vgg(eeg_train_data,fnirs_train_data,resnet):
    inputE = Input(shape=(200, 32,3))
    inputF = Input(shape=(32, 72, 3))

    normalization_layerE = Normalization()
    normalization_layerF = Normalization()
    normalization_layerE.adapt(eeg_train_data)   
    normalization_layerF.adapt(fnirs_train_data)     

    norm_layerE = normalization_layerE(inputE)
    norm_layerF = normalization_layerF(inputF)

    if resnet:
      baseline_modele= tf.keras.applications.ResNet50(include_top=False,input_shape=(200, 32,3), classes=2, weights=None)
    else:
      baseline_modele= tf.keras.applications.VGG19(include_top=False,input_shape=(200, 32,3),weights=None) 

    baseline_modelf =vgg16.VGG16(weights=None, include_top=False, input_shape=(32, 72, 3))

    # the first branch operates on the first input: EEG
    e = (baseline_modele)(norm_layerE)
    e = Flatten()(e)
    e = Dense(128, activation='relu')(e)
    e = Model(inputs=inputE, outputs=e)

    # the second branch operates on the second input: FNIRS
    f = (baseline_modelf)(norm_layerF)
    f = Flatten()(f)
    f = Dense(128, activation='relu')(f)
    f = Model(inputs=inputF, outputs=f)
    # combine the output of the two branches
    combined = concatenate([e.output, f.output])

    #MLP
    z = Dense(256, activation="relu")(combined)
    z = Dense(64, activation="relu")(z)
    z = Dense(1, activation="sigmoid")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[e.input, f.input], outputs=z)
    return model

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import transformer
import keras
import tensorflow as tf 
from keras.callbacks import ModelCheckpoint
from keras import layers


#from models import LSTMMultiClass, TransformerClassifier, LSTMBinary, CNN_1D, CNN_1D_multihead

training = True

balanced_dataset = True
binary_classification = False

current_action = 'ASSEMBLY1'

def oneVsAll(labels, int_labels, label):
    '''Converts a multi-class problem into a binary problem by setting all labels'''
    unique_labels = np.unique(labels)
    print(unique_labels)
    int_label = np.argwhere(unique_labels == label)[0][0]
    mask = int_labels == int_label
    int_labels[mask] = 0
    int_labels[~mask] = 1
    return int_labels

def ignore_features(dataset):
    '''Excludes gyroscope data from the dataset'''
    mask = np.ones((24), dtype=bool)
    mask[[3,4,5,9,10,11,15,16,17,21,22,23]] = False
    dataset = dataset[:, :, mask]
    return dataset



def get_accuracy(pred, test):
    '''Returns the accuracy of the model on the multiclass problem'''
    correct = 0
    wrong = 0
    for p, t in zip(np.argmax(pred,1), test):
        if p == t:
            correct+=1
        else:
            wrong+=1
    return (correct/test.shape[0])*100

def normalize(data):
    '''Normalizes the data by dividing each feature by its maximum value'''
    maxes = np.amax(data, axis=(0,1))
    # mins = np.amin(data, axis=(0,1))
    # return (2*(data-mins)/(maxes-mins))-1
    return data/maxes


def full_scale_normalize(data):
    acceleration_idxs = [0,1,2,6,7,8,12,13,14,18,19,20]
    gyroscope_idxs = [3,4,5,9,10,11,15,16,17,21,22,23]

    # 1g equals 8192. The full range is 2g
    data[:,:,acceleration_idxs] = data[:,:,acceleration_idxs] / 16384.0
    data[:,:,gyroscope_idxs] = data[:,:,gyroscope_idxs] / 1000.0

    return data


def add_feature_profiles(dataset):
    '''Adds the module of the 3D vector of each feature to the dataset'''
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], -1, 3)
    module = np.sqrt(np.sum(dataset**2, axis=3))
    dataset = np.concatenate((dataset, module[..., None]), axis=3)
    return dataset.reshape(dataset.shape[0], dataset.shape[1], -1)


print("\n--- Data Loading ---")

if balanced_dataset:
    print("\n--- Loading Balanced Dataset ---")
    train_dataset = np.load('balanced_datasets/train_balanced_data(6750_500_24).npy').astype('float32')
    train_labels = np.load('balanced_datasets/train_balanced_labels(6750_1).npy', allow_pickle=True)#.astype('int32')

    # test_dataset = np.load('balanced_datasets/train_balanced_data.npy').astype('float32')
    # test_labels = np.load('balanced_datasets/train_balanced_labels.npy', allow_pickle=True)#.astype('int32')

    test_dataset = np.load('test_data_shape(1233_500_24).npy').astype('float32')
    test_labels = np.load('test_labels_shape(1233_1).npy')

else:
    print("\n--- Loading Unbalanced Dataset ---")
    train_dataset = np.load('train_data_shape(4950_500_24).npy').astype('float32')
    # dataset = torch.load('filename')
    train_labels = np.load('train_labels_shape(4950_1).npy')
    test_dataset = np.load('test_data_shape(1233_500_24).npy').astype('float32')
    # dataset = torch.load('filename')
    test_labels = np.load('test_labels_shape(1233_1).npy')


unique_labels = np.unique(train_labels)
print(unique_labels)

# Convert string labels to integer labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels.ravel())
test_labels = label_encoder.fit_transform(test_labels.ravel())
# if balanced_dataset:
#     integer_labels = labels.ravel()


if binary_classification:
    print("\n--- Reducing to a Binary Classification Problem ---")
    integer_labels = oneVsAll(labels, integer_labels, current_action)

# if balanced_dataset:
#     unique_labels = np.unique(integer_labels)


# dataset = ignore_features(dataset)
# dataset = add_feature_profiles(dataset)
# dataset = dataset[:,:,12:16]

print("Loaded dataset and labels: ")
# print(f'\t{dataset.shape=}')
print('TRAIN')
# print(f'\t{train_labels.shape=}')
print("Most populated class: ", unique_labels[np.argmax(np.bincount(test_labels))])
print('TEST')
# print(f'\t{test_labels.shape=}')
print("Most populated class: ", unique_labels[np.argmax(np.bincount(test_labels))])



train_dataset = full_scale_normalize(train_dataset)
test_dataset = full_scale_normalize(test_dataset)  
print("\nSplitted dataset and labels: ")
print(f'\t{train_dataset.shape=}')
print(f'\t{test_dataset.shape=}')
print(f'\t{train_labels.shape=}')
print(f'\t{test_labels.shape=}')

print("\nSplitted dataset and labels: ")
print(f'\t{train_dataset.shape=}')
print(f'\t{test_dataset.shape=}')
print(f'\t{train_labels.shape=}')
print(f'\t{test_labels.shape=}')

print("\n--- Training ---")

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# print(f'\nSetting torch device to: {sess=}')
# # print(device_lib.list_local_devices())
# print(tf.test.gpu_device_name())
# # tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# # config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
# # sess = tf.Session(config=config) 
# keras.backend.set_session(sess)
# exit()
# Set device to CUDA if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# print(f'\nSetting torch device to: {device=}')

# x = torch.Tensor(train_dataset).to(device)
# y = torch.Tensor(train_labels).squeeze().long().to(device)
x = train_dataset
y = train_labels
x_test = test_dataset
y_test = test_labels
# x_test = torch.Tensor(test_dataset).to(device)
# y_test = torch.Tensor(test_labels).squeeze().long().to(device)
print(f'{x.shape=}')
print(f'{y.shape=}')
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
print(f'{x_train.shape=}')
print(f'{y_train.shape=}')
print(f'{x_val.shape=}')
print(f'{y_val.shape=}')

# Define hyperparameters
input_dim = train_dataset[0].shape[-1]
hidden_dim = 8
n_layers = 2
if binary_classification:
    output_dim = 1
else:
    output_dim = unique_labels
'''multihead cnn works best with 0.0005'''
'''singlehead cnn works best with 0.0001'''

lr = 0.00005
epochs = 200
batch_size = 32
dropout = 0.5
l2_lambda = 0.0001



# Set up early stopping
patience = 8
best_val_loss = float('inf')
counter = 0

print(f'\nHyperparameters: ')
print(f'\t{input_dim=}')
print(f'\t{hidden_dim=}')
print(f'\t{output_dim=}')
print(f'\t{lr=}')
print(f'\t{epochs=}')
print(f'\t{batch_size=}\n')
print(f'\t{l2_lambda=}\n')
print(f'\t{patience=}\n')

input_shape = x.shape[1:]
print(f'{unique_labels.shape[0]=}')
model = transformer.build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128,64],
    mlp_dropout=0.4,
    dropout=0.2,
    n_classes=unique_labels.shape[0]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()
callbacks = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True), 
             ModelCheckpoint('transformer.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

if training:
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

else: 
    model = keras.models.load_model('transformer.h5')

model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test)

if binary_classification:
    metric = BinaryAccuracy(device=torch.device('cuda'))
    metric.update(y_pred.squeeze().cuda(), y_test.cuda())
    acc = metric.compute()
else:
    acc = get_accuracy(y_pred, y_test)
print("accuracy: ", acc)


# Build confusion matrix
if binary_classification:
    y_pred = y_pred>0.5
    y_pred = y_pred
    cf_matrix = confusion_matrix(y_pred, y_test)
else:
    cf_matrix = confusion_matrix(np.argmax(y_pred,1), y_test)
print(cf_matrix)
print(np.sum(cf_matrix, axis=1)[:, None])
cf_matrix = np.around(cf_matrix / np.sum(cf_matrix, axis=1)[:, None] * 100, decimals=1)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in unique_labels],
                     columns = [i for i in unique_labels])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
if binary_classification:
    plt.savefig(f'binary_models/{current_action}_cf.png')
else:
    plt.savefig('output.png')

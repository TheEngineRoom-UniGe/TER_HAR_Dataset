import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
import transformer
import keras
import tensorflow as tf 
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib



from models import LSTMMultiClass, TransformerClassifier, LSTMBinary, CNN_1D, CNN_1D_multihead

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


def add_feature_profiles(dataset):
    '''Adds the module of the 3D vector of each feature to the dataset'''
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], -1, 3)
    module = np.sqrt(np.sum(dataset**2, axis=3))
    dataset = np.concatenate((dataset, module[..., None]), axis=3)
    return dataset.reshape(dataset.shape[0], dataset.shape[1], -1)



print("\n--- Data Loading ---")

if balanced_dataset:
    print("\n--- Loading Balanced Dataset ---")
    dataset = np.load('balanced_datasets/balanced_data1.npy').astype('float32')
    # dataset = torch.load('filename')
    labels = np.load('balanced_datasets/balanced_labels1.npy', allow_pickle=True)#.astype('int32')

else:
    print("\n--- Loading Unbalanced Dataset ---")
    dataset = np.load('data_shape(2699_2981_24).npy').astype('float32')
    # dataset = torch.load('filename')
    labels = np.load('labels_shape(2699_1).npy')


# Convert string labels to integer labels
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels.ravel())
# if balanced_dataset:
#     integer_labels = labels.ravel()


if binary_classification:
    print("\n--- Reducing to a Binary Classification Problem ---")
    integer_labels = oneVsAll(labels, integer_labels, current_action)
unique_labels = np.unique(labels)
# if balanced_dataset:
#     unique_labels = np.unique(integer_labels)


# dataset = ignore_features(dataset)
# dataset = add_feature_profiles(dataset)
# dataset = dataset[:,:,12:16]

print("Loaded dataset and labels: ")
print(f'\t{dataset.shape=}')
print(f'\t{integer_labels.shape=}')
print("Most populated class: ", np.argmax(np.bincount(integer_labels)))


# Split the data into training and test sets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, integer_labels, test_size=0.2, random_state=42)

train_dataset = normalize(train_dataset)
test_dataset = normalize(test_dataset)  

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
print(f'\t{x.shape=}')
print(f'\t{y.shape=}')

# Define hyperparameters
input_dim = dataset[0].shape[-1]
hidden_dim = 8
n_layers = 2
if binary_classification:
    output_dim = 1
else:
    output_dim = unique_labels
'''multihead cnn works best with 0.0005'''
'''singlehead cnn works best with 0.0001'''

lr = 0.0005
epochs = 200
batch_size = 8
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
    mlp_units=[128],
    mlp_dropout=0.5,
    dropout=0.5,
    n_classes=unique_labels.shape[0]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()
callbacks = [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True), 
             ModelCheckpoint('transformer.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

model.fit(
    x,
    y,
    validation_split=0.2,
    epochs=200,
    batch_size=2,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)

if binary_classification:
    model.load_state_dict(keras.load(f"binary_models/{current_action}.pth"))
else:
    model.load_state_dict(keras.load("transformer.h5"))
model.eval()
x_test = x_test
y_pred = model(x_test)  
# if binary_classification:
    # metric = BinaryAccuracy(device=torch.device('cuda'))
    # metric.update(y_pred.squeeze().cuda(), y_test.cuda())
    # acc = metric.compute()
# else:
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

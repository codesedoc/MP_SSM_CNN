import matplotlib.pyplot as plt
import utils.file_tool as file_tool
import numpy as np
import colorsys

def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


color_list = []
with_test_data = file_tool.load_data_pickle('with_test_data.pkl')
without_test_data = file_tool.load_data_pickle('without_test_data.pkl')

def show_data(data, feature_index = 1):
    momentum_count = len(data, )
    colors = get_colors(momentum_count)
    # rgbs = np.arange(0,momentum_count)/momentum_count
    # rgbs = rgbs.reshape(momentum_count,1)
    # rgbs = np.broadcast_to(rgbs, (momentum_count,3))
    for momentum_data, color in zip(data, colors):
        momentum = momentum_data['momentum']

        records = momentum_data['records']
        epochs = records [..., 0]
        feature = records[..., feature_index]
        plt.plot(epochs, feature, color = color ,linewidth = 1.0, label = 'momentum:'+str(momentum))

        max_value = max(feature.tolist())
        max_index = feature.tolist().index(max_value)
        plt.plot(max_index + 1, max_value, 'r*')
        plt.annotate(str(round(max_value,4)), xy=(max_index+1, max_value), xytext=(50, 50),
                    xycoords='data', textcoords='offset pixels',
                    arrowprops=dict(facecolor='black', arrowstyle='->')
                    )
    plt.legend()

plt.figure(1, figsize=(14,14))

plt.subplot(311)
plt.title('with_test')
plt.ylabel('train_accuracy')
show_data(with_test_data, 1)
plt.subplot(312)
plt.ylabel('train_loss')
show_data(with_test_data, 2)
plt.subplot(313)
plt.ylabel('test_accuracy')
show_data(with_test_data, 3)
plt.xlabel('epoch')

plt.figure(2, figsize=(14,14))

plt.subplot(211)
plt.title('without_test')
plt.ylabel('train_accuracy')
show_data(without_test_data, 1)
plt.subplot(212)
plt.ylabel('train_loss')
show_data(without_test_data, 2)
plt.xlabel('epoch')
plt.show()


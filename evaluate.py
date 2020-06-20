# Visualize the Results

import matplotlib.pyplot as plt

assert len(train_loss_list) and len(val_loss_list)

epochs_list = list(range(0, len(train_loss_list)))
epochs_labels = list(range(1, len(train_loss_list)+1))


plt.figure()
plt.plot(epochs_list, train_loss_list, label = 'Training Loss', color='black')
plt.plot(epochs_list,val_loss_list, label = 'Validation Loss', color='magenta')
plt.legend(loc=1) 

plt.grid(True)
# plt.xticks(epochs_list, epochs_labels)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Focal Loss on Training and Validation Sets')

plt.figure()
plt.plot(epochs_list, val_iou_building_list, label='IoU Building', color='red')
plt.plot(epochs_list, val_iou_road_list, label='IoU Road', color='blue')
plt.legend(loc=4)

plt.grid(True)
# plt.xticks(epochs_list, epochs_labels)
plt.xlabel('Epoch')
plt.ylabel('Intersection over Union (IoU)')
plt.title('IoU on Validation Set')

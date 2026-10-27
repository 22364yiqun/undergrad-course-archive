import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def acc_loss_curve(train_loss, train_acc, val_loss, val_acc, line_path):  # line_path 保存路径
    # train_loss, train_acc, val_loss, val__acc
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.plot(train_acc, label='Train_Accuracy')
    plt.plot(val_acc, label='Validation_Accuracy')
    plt.xlabel('Num of Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.rcParams.update({'font.size': 15})
    
    plt.subplot(122)
    plt.plot(train_loss, label='Train_Loss')
    plt.plot(val_loss, label='Validation_Loss')
    plt.xlabel('Num of Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(line_path, dpi=400, format='png')
    plt.close()

def plot_heatmap(matrix, label_class, plot_path):
    plt.imshow(matrix, cmap=plt.cm.Blues)
    plt.title("Normalized confusion matrix")  # title
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")

    num_classes = len(label_class)
    labels_name = list(label_class.keys())
    plt.yticks(range(num_classes), labels_name)  # y轴标签
    plt.xticks(range(num_classes), labels_name)  # x轴标签

    for x in range(num_classes):
        for y in range(num_classes):
            value = float(matrix[y, x])  # 数值处理
            plt.text(x, y, value, fontsize=10, verticalalignment='center', horizontalalignment='center')  # 写值

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    plt.colorbar()  # 色条
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def result_record(labels, pred_labels, **option):
    C2 = confusion_matrix(labels, pred_labels)

    f1 = f1_score(labels, pred_labels, average='macro')
    recall = recall_score(labels, pred_labels, average='macro')
    precision = precision_score(labels, pred_labels, average='macro')
    accuracy = accuracy_score(labels, pred_labels)

    plot_heatmap(C2, option['label_class'], option['matrix_path'])

    class_measure = classification_report(labels, pred_labels)
    
    # Record the results
    with open(option['log_path'], 'a+') as fout:
        model_file_name_n = option['model_path'].split('/')[-1]
        fout.write("---------------------------------------------------------------------------------------------\n")
        fout.write('\nModel Test\n')
        fout.write("---------------------------------------------------------------------------------------------\n")
        fout.write(f'{model_file_name_n.upper()}\nBest Model | Test Acc: {accuracy*100:.2f}% | Test F1: {f1*100:.2f}%\n')
        print('\n************************ Model Test ****************************')
        print(f'{model_file_name_n.upper()}\nBest Model | Test Acc: {accuracy*100:.2f}% | Test F1: {f1*100:.2f}%')
        fout.write("---------------------------------------------------------------------------------------------\n")
        fout.write('\nClassification Report\n')
        fout.write("---------------------------------------------------------------------------------------------\n")
        fout.write(class_measure)
        print('\n******************* Classification Report ***********************')
        print(class_measure, end='')
        fout.write("---------------------------------------------------------------------------------------------\n")
        fout.write('\nTest Metrics\n')
        print('\n************************ Test Metrics ***************************')
        fout.write("---------------------------------------------------------------------------------------------\n")        
        print(f'f1 score: {f1*100:.2f}%\nrecall score: {recall*100:.2f}%\nprecision score: {precision*100:.2f}%\naccuracy score: {accuracy*100:.2f}%')
        fout.write(f'f1 score: {f1*100:.2f}%\nrecall score: {recall*100:.2f}%\nprecision score: {precision*100:.2f}%\naccuracy score: {accuracy*100:.2f}%\n')
        fout.write("---------------------------------------------------------------------------------------------\n")        
        fout.write('\nConfusion Matrix\n')
        fout.write("---------------------------------------------------------------------------------------------\n")        
        print('\n********************* Confusion Matrix *************************')
        print(C2)
        np.savetxt(fout, C2, fmt='%f', delimiter=', ')
        fout.write("---------------------------------------------------------------------------------------------\n")
    
    return accuracy, f1, recall, precision


def final_record(test_acc_list, test_f1_list, test_recall_list, test_precision_list, **opt):
    if opt['verbose']:
        print('test_acc_list: ', test_acc_list)
        print(f'test_acc_avg: {np.mean(test_acc_list)*100:.2f}%')
        print(f'test_acc_max: {np.max(test_acc_list)*100:.2f}%')
        print(f'test_acc_std: {np.std(test_acc_list):.2f}')
        print('test_acc_max_index: ', np.argmax(test_acc_list) + 1)

        print()
        print('test_f1_list: ', test_f1_list)
        print(f'test_f1_avg: {np.mean(test_f1_list)*100:.2f}%')
        print(f'test_f1_max: {np.max(test_f1_list)*100:.2f}%')
        print(f'test_f1_std: {np.std(test_f1_list):.2f}')
        print('test_f1_max_index: ', np.argmax(test_f1_list) + 1)

        print()
        print('test_recall_list: ', test_recall_list)
        print(f'test_recall_avg: {np.mean(test_recall_list)*100:.2f}%')
        print(f'test_recall_max: {np.max(test_recall_list)*100:.2f}%')
        print(f'test_recall_std: {np.std(test_recall_list):.2f}')
        print('test_recall_max_index: ', np.argmax(test_recall_list) + 1)

        print()
        print('test_precision_list: ', test_precision_list)
        print(f'test_precision_avg: {np.mean(test_precision_list)*100:.2f}%')
        print(f'test_precision_max: {np.max(test_precision_list)*100:.2f}%')
        print(f'test_precision_std: {np.std(test_precision_list):.2f}')
        print('test_precision_max_index: ', np.argmax(test_precision_list) + 1)

    acc_path = f"{opt['root_folder']}/{opt['log_folder']}/{opt['log_file_name']}_acc.txt"

    with open(acc_path, 'a+') as f:
        f.write('test_acc_list: \n')
        f.write(str(test_acc_list) + '\n')
        f.write(f'test_acc_avg: {np.mean(test_acc_list)*100:.2f}\n')
        f.write(f'test_acc_std: {np.std(test_acc_list)*100:.2f}\n')
        f.write(f'test_acc_max: {np.max(test_acc_list)*100:.2f}\ntest_acc_max_index: {np.argmax(test_acc_list) + 1}\n')
        
        f.write('\ntest_f1_list: \n')
        f.write(str(test_f1_list) + '\n')
        f.write(f'test_f1_avg: {np.mean(test_f1_list)*100:.2f}\n')
        f.write(f'test_f1_std: {np.std(test_f1_list)*100:.2f}\n')
        f.write(f'test_f1_max: {np.max(test_f1_list)*100:.2f}\ntest_f1_max_index: {np.argmax(test_f1_list) + 1}\n')

        f.write('\ntest_recall_list: \n')
        f.write(str(test_recall_list) + '\n')
        f.write(f'test_recall_avg: {np.mean(test_recall_list)*100:.2f}\n')
        f.write(f'test_recall_std: {np.std(test_recall_list)*100:.2f}\n')
        f.write(f'test_recall_max: {np.max(test_recall_list)*100:.2f}\ntest_recall_max_index: {np.argmax(test_recall_list) + 1}\n')

        f.write('\ntest_precision_list: \n')
        f.write(str(test_precision_list) + '\n')
        f.write(f'test_precision_avg: {np.mean(test_precision_list)*100:.2f}\n')
        f.write(f'test_precision_std: {np.std(test_precision_list)*100:.2f}\n')
        f.write(f'test_precision_max: {np.max(test_precision_list)*100:.2f}\ntest_precision_max_index: {np.argmax(test_precision_list) + 1}\n')

    pass

def tsne_show(feature, label, plot_path):
    # fault_label = []
    # for label in label.tolist():
    #     fault_label.append(label_class[label])

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_features = tsne.fit_transform(feature)

    unique = np.unique(label)
    colors = [plt.cm.rainbow(i / float(len(unique)-1)) for i in range(len(unique))]

    for i, u in enumerate(unique):
        # xi = [tsne_features[:,0][j] for j in range(len(tsne_features[:,0])) if label[j]==u]
        # yi = [tsne_features[:,1][j] for j in range(len(tsne_features[:,0])) if label[j]==u]
        xi = [tsne_features[j, 0] for j, lbl in enumerate(label) if lbl == u]
        yi = [tsne_features[j, 1] for j, lbl in enumerate(label) if lbl == u]
        plt.scatter(xi, yi, c=colors[i], s=7, alpha=1, label=str(u))

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc=8, ncol=len(unique)//2, handletextpad=0.1,\
        scatterpoints=1, labelspacing=0.1, columnspacing=0.1)
    plt.xticks([])
    plt.yticks([])

    # plt.scatter(tsne_features[:,0], tsne_features[:,1], c=label.tolist(), cmap='Pastel1', marker='o')
    # sns.scatterplot(tsne_features[:,0], tsne_features[:,1], hue=fault_label, palette=sns.color_palette('pastel', len(set(fault_label))))
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
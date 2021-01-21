import torch
import gc
import numpy as np
# import EarlyStopping
from train.pytorchtools import EarlyStopping
from sklearn.metrics import balanced_accuracy_score

from tqdm import tqdm

from utils.tensorboard import init_writer, save_graph, save_weigths, save_grads, save_video_inputs, save_confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def train_epoch(device, trainloader, model, optimizer, is_mediaeval):
    # Metrics
    running_train_correct = 0.0
    running_train_total = 0.0
    running_loss = 0.0
    balanced_accuracy = 0.0
    # Set train mode
    model.train()
    with tqdm(total=len(trainloader)) as t:
        for i, batch in enumerate(trainloader):
            # Get batch
            inputs, labels = batch[0].cuda(), batch[1].cuda()
            res_size = 0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())[0]

            if isinstance(model, torch.nn.DataParallel):
                loss = model.module.loss(outputs, labels.long())
            else:
                loss = model.loss(outputs, labels.long())
            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if type(outputs) is tuple:
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            if is_mediaeval:
                balanced_accuracy = balanced_accuracy_score(labels.cpu(),predicted.cpu())

                metrics = {"loss": running_loss / (i + 1),
                        "balanced acc": balanced_accuracy }
            else:

                running_train_total += labels.size(0)
                running_train_correct += (predicted == labels.long()).sum().item()

                metrics = {"loss": running_loss / (i + 1),
                        "acc": 100 * running_train_correct / running_train_total}

            # Update tqdm
            t.set_description('Batch %i' % (i + 1))
            t.set_postfix(**metrics)
            t.update()

            # gc.collect()

    return metrics


def qualitative_analysis(model, device, testLoader):
    correct = 0
    total = 0

    predicted_list = []
    labels_list = []
    test_losses = []
    # outputs=None
    best_weights = None
    best_score = 0

    with torch.no_grad():
        # Set evaluation mode
        model.eval()
        for idx, data in enumerate(testLoader):
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs, weights = model(inputs.float())

            _, predicted = torch.max(outputs.data, 1)

            # if (predicted != labels.long() and predicted == 0 and outputs.data[0][1] > best_score):
            #     print(idx, outputs.data[0][1])
            #     best_score = outputs.data[0][1]
                
            # if idx == 33:
            if idx == 35:
                print(predicted, labels.long())
                best_weights = weights
                break
    
    best_weights = best_weights.cpu().numpy()
    weights = np.reshape(best_weights, (60, 1))
    # weights = np.reshape(best_weights, (150,1))
    indexes = np.arange(60)

    # weights = np.insert(weights,1,indexes,axis=1)
    weights = np.insert(weights, 1, indexes, axis=1)

    print("saving attention weights ...")
    np.savetxt("attention_weights/soft_att_incorrect.csv", weights, delimiter=",")
    
    # RWF
    # best_weights = best_weights.cpu().numpy()
    # weights = np.reshape(best_weights, (150, 3))
    # # weights = np.reshape(best_weights, (150,1))
    # indexes = np.arange(150)

    # # weights = np.insert(weights,1,indexes,axis=1)
    # weights = np.insert(weights, 1, indexes, axis=1)

    # print("saving attention weights ...")
    # np.savetxt("clusters_multimodal_incorrect.csv", weights, delimiter=",")


def test_epoch(device, testloader, model, early_stop, cv_idx):
    correct = 0
    total = 0

    predicted_list = []
    labels_list = []
    test_losses = []

    with torch.no_grad():
        # Set evaluation mode
        model.eval()
        for idx, data in enumerate(testloader):
            inputs, labels = data[0].cuda(), data[1].cuda()

            outputs = model(inputs.float())[0]
            if isinstance(model, torch.nn.DataParallel):
                loss = model.module.loss(outputs, labels.long())
            else:
                loss = model.loss(outputs, labels.long())

            if type(outputs) is tuple:
                _, predicted = torch.max(outputs[0].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

            predicted_list.append(predicted.cpu())
            labels_list.append(labels.cpu())
            test_losses.append(loss.item())

    test_losses = np.average(test_losses)
    print("test loss : {}".format(test_losses))

    early_stop(100 * correct / total, model, cv_idx)

    metrics = {'acc': 100 * correct / total,
               'predicted': predicted_list, 'labels': labels_list}
    print('Accuracy on test dataset: %f %%' % (metrics["acc"]))

    return metrics, test_losses

def test_mediaeval(device, testloader, model, model_name, if_att, att_type, official_metric=True):
    correct = 0
    total = 0

    predicted_list = []
    labels_list = []        
    test_losses = []
    preds = []

    if official_metric:
        if model_name == 'att_clusters' or model_name == 'att_clusters_multimodal' or model_name == 'att_clusters_late':
            test_predictions_mediaeval = open("mediaeval_2015_predictions_{}.txt".format(model_name),"w")
        elif model_name == 'lstm_late':
            if if_att:
                test_predictions_mediaeval = open("mediaeval_2015_predictions_{}_{}_late_fusion.txt".format(model_name, att_type),"w")
            else:
                test_predictions_mediaeval = open("mediaeval_2015_predictions_{}_{}_late_fusion.txt".format(model_name, "last_seg"),"w")
        elif model_name == 'lstm':
            if if_att:
                test_predictions_mediaeval = open("mediaeval_stuffs/mediaeval_2015_predictions_{}_{}.txt".format(model_name, att_type),"w")
            else:
                test_predictions_mediaeval = open("mediaeval_stuffs/mediaeval_2015_predictions_{}_{}.txt".format(model_name, "last_seg"),"w")

        test_filenames_mediaeval = open("mediaeval_stuffs/test_mediaeval_raw.txt", "r")

        with torch.no_grad():
            # Set evaluation mode
            model.eval()
            video_names = test_filenames_mediaeval.readlines()
            test_filenames_mediaeval.close()
            for idx, data in enumerate(testloader):
                inputs, labels = data[0].cuda(), data[1].cuda()

                outputs = model(inputs.float())[0]
                if isinstance(model, torch.nn.DataParallel):
                    loss = model.module.loss(outputs, labels.long())
                else:
                    loss = model.loss(outputs, labels.long())

                if type(outputs) is tuple:
                    _, predicted = torch.max(outputs[0].data, 1)
                    confidence = outputs.data[:,1]
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    confidence = outputs.data[:,1]
                total += labels.size(0) 
                correct += (predicted == labels.long()).sum().item()
                
                if predicted.cpu().numpy()[0] == 1:
                    label_predicted = 't'
                else:
                    label_predicted = 'f'
                
                video_name = video_names[idx]
                test_predictions_mediaeval.write(video_name.split("\n")[0] + str(confidence.cpu().numpy()[0]) + " " + label_predicted + " \n")

                predicted_list.append(predicted.cpu())
                labels_list.append(labels.cpu())
                test_losses.append(loss.item())

        test_losses = np.average(test_losses)
        print("test loss : {}".format(test_losses))

        test_predictions_mediaeval.close()
    else:
        with torch.no_grad():
            # Set evaluation mode
            model.eval()
            for idx, data in enumerate(testloader):
                inputs, labels = data[0].cuda(), data[1].cuda()

                outputs = model(inputs.float())[0]
                if isinstance(model, torch.nn.DataParallel):
                    loss = model.module.loss(outputs, labels.long())
                else:
                    loss = model.loss(outputs, labels.long())

                if type(outputs) is tuple:
                    _, predicted = torch.max(outputs[0].data, 1)
                    confidence = outputs.data[:,1]
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    confidence = outputs.data[:,1]
                # total += labels.size(0) 
                preds.append(predicted.cpu())
                labels_list.append(labels.cpu())

                test_losses.append(loss.item())

        balanced_accuracy = balanced_accuracy_score(labels_list,preds)
        test_losses = np.average(test_losses)
        if  model_name == 'lstm':
            if if_att:
                print("Balanced accuracy [{}-{}]: {}".format(model_name, att_type, balanced_accuracy))
            else:
                print("Balanced accuracy [{}-{}]: {}".format(model_name, 'last_seg', balanced_accuracy))
        else:
            print("Balanced accuracy [{}]: {}".format(model_name, balanced_accuracy))
        print("Test loss : {}".format(test_losses))

    # if model_name == 'att_clusters' or model_name == 'att_clusters_multimodal' or model_name == 'att_clusters_late':

def train(model, optimizer, n_epochs, device, trainloader, testloader, model_dir, log_dir, cv_idx=1, is_mediaeval=False, ini_epoch=0, test=True):

    pattern = "\n\nEpoch: %0{0}d/%0{0}d".format(len(str(n_epochs)))

    tb_writer = init_writer(log_dir)

    # trainloader.dataset.save_videos(tb_writer, mode="train")
    # testloader.dataset.save_videos(tb_writer, mode="test")
    train_metrics = {}
    test_metrics = {}
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(ini_epoch, n_epochs):  # loop over the dataset multiple times
        print(pattern % ((epoch + 1), n_epochs))

        train_metrics = train_epoch(device, trainloader, model, optimizer,is_mediaeval)
        for k, v in train_metrics.items():
            if type(v) is not dict:
                tb_writer.add_scalar('train/{}'.format(k), v, epoch)

        if test:
            test_metrics, test_losses = test_epoch(
                device, testloader, model, early_stopping, cv_idx=cv_idx)
            test_metrics['acc'] = early_stopping.best_score
            # test_metrics, test_losses = test_epoch(device, testloader, model)
            for k, v in test_metrics.items():
                if type(v) is not dict and type(v) is not list:
                    tb_writer.add_scalar('test/{}'.format(k), v, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Saving model
        # print("Saving model in:", "{}/checkpoint.pth".format(model_dir))
        # state = {'epoch': epoch + 1,
        #          'state_dict': model.module.state_dict()
        #          if isinstance(model, torch.nn.DataParallel)
        #          else model.state_dict(),
        #          'optimizer': optimizer.state_dict()}

        # torch.save(state, "{}/checkpoint.pth".format(model_dir))
        torch.save(model.state_dict(), "checkpoint.pth")

        # if hasattr(model, 'rcnn') and  hasattr(model.rcnn, 'alpha'):
        #     for i, alpha in enumerate(model.rcnn.alpha):
        #         print("coefficients i:", i, torch.squeeze(alpha))

        # if epoch % 10 == 0:
        #     save_weigths(model, tb_writer, epoch)
        #     # save_grads(train_metrics['grads'], tb_writer, epoch)
        #
        #     if test:
        #         correct_labels = torch.cat(test_metrics['labels'])
        #         predict_labels = torch.cat(test_metrics['predicted'])
        # save_confusion_matrix(correct_labels,
        #                       predict_labels,
        #                       labels=trainloader.dataset.classes,
        #                       normalize=False,
        #                       title='Confusion_matrix/Test',
        #                       tb_writer=tb_writer,
        #                       epoch=epoch)

        # save_confusion_matrix(correct_labels,
        #                       predict_labels,
        #                       labels=trainloader.dataset.classes,
        #                       normalize=True,
        #                       title='Confusion_matrix/Test-Normalized',
        #                       tb_writer=tb_writer,
        #                       epoch=epoch)

    if test:
        metrics = {"train": train_metrics, "test": test_metrics}
    else:
        metrics = {"train": train_metrics}

    print('Finished Training')

    print("train finished log path:", log_dir)

    # Saving final confusion matrix
    # if test:
    #     correct_labels = torch.cat(test_metrics['labels'])
    #     predict_labels = torch.cat(test_metrics['predicted'])
    #     save_confusion_matrix(correct_labels,
    #                           predict_labels,
    #                           labels=trainloader.dataset.classes,
    #                           normalize=False,
    #                           title='Confusion_matrix/Test-final',
    #                           tb_writer=tb_writer,
    #                           epoch=epoch)
    #
    #     save_confusion_matrix(correct_labels,
    #                           predict_labels,
    #                           labels=trainloader.dataset.classes,
    #                           normalize=True,
    #                           title='Confusion_matrix/Test-Normalized-final',
    #                           tb_writer=tb_writer,
    #                           epoch=epoch)

    tb_writer.close()

    return metrics

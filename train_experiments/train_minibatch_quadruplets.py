import os
import time
from collections import Counter

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from datasets.BatchSampler import BalancedBatchSampler
from datasets.cluster_dataset import ClusterDataset
from datasets.dataholder import DataHolder
from losses.batch_all_triplet_with_clusters_loss import BatchAllTripletWithClustersLossSemiHard
from models.TripletNet import EmbeddingNet
from utils import parse_configuration, test_knn_classifier, plot_metrics, plot_conf_matrix, stratifiedKCrossVKnn, \
    stratifiedKCrossVDecisionTree, test_tree_classifier
from utils.dataset_utils import initialize_datasets
from utils.visualizer import Visualizer


def train_minibatch_quadruplets(config_file):
    print('Reading config file: ' + str(config_file))
    configuration = parse_configuration(config_file)

    print('Reading dataset: ' + configuration['dataset']['dataset_name'])
    data_holder = DataHolder(dataset_path=configuration['dataset']['dataset_path'],
                             majority_labels=configuration['dataset']['majority_classes'],
                             undersample=configuration['dataset']['undersample'] == "True",
                             k_param=configuration['clustering']['k'],
                             undersample_method=configuration['dataset']['undersample_method'],
                             dataset_name=configuration['dataset']['dataset_name']
                             )
    print("Initializing datasets")

    train_dataset, test_dataset, whole_dataset = initialize_datasets(data_holder)

    batch_size = configuration['model']['batch_size']
    n_classes = np.unique(data_holder.labels).size

    # Batch Sampler seems to be critical, when handling online triplet loss

    train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=n_classes,
                                               n_samples=max(1, batch_size // n_classes), name="train")

    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=n_classes,
                                              n_samples=max(1, batch_size // n_classes), name="test")

    trainClusterDataset = ClusterDataset(train_dataset.train_data, train_dataset.train_labels,
                                         data_holder.train_clusters)
    testClusterDataset = ClusterDataset(test_dataset.test_data, test_dataset.test_labels, data_holder.test_clusters)

    train_dataset_loader = DataLoader(trainClusterDataset, batch_sampler=train_batch_sampler)
    test_dataset_loader = DataLoader(testClusterDataset, batch_sampler=test_batch_sampler)

    embedding_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    embedding_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    whole_dataset_loader = DataLoader(whole_dataset, batch_size=batch_size * 2)

    if configuration['weights_enabled'] == "True":
        print("Initializing weights")
        train_weights = data_holder.train_weights
        test_weights = data_holder.test_weights
    else:
        train_weights = None
        test_weights = None

    print('Initializing model')
    model = EmbeddingNet(configuration['model'])

    starting_epoch = configuration['model_params']['start_epoch']
    num_epochs = configuration['model']['epochs']

    # Load model params if necessary
    if configuration['model_params']['load_params'] == "True":
        model.load_state_dict(torch.load(os.path.join(configuration['model_params']['saved_model_path'])))

    print("Initializing optimizer")
    optimizer = Adam(params=model.parameters(), lr=configuration['optimizer']['learning_rate'])

    # Scheduler seems to be critical, when handling online triplet loss
    print("Initializing scheduler")
    scheduler = StepLR(optimizer, step_size=configuration['scheduler']['step_size'],
                       gamma=configuration['scheduler']['gamma'])

    # Initializing Visualiser
    vis = Visualizer()
    vis.reduce_data_to_2d_with_pca(data_holder.data, data_holder.labels,
                                   os.path.join(configuration['checkpoint_path'], 'images'),
                                   os.path.join(configuration['checkpoint_path'],
                                                'images/{}_original.png'.format(
                                                    configuration['save_name'])), 'Oryginalny zbiór danych')

    # Initalize dataset to caluclate distances between samples
    freq = Counter(data_holder.labels)
    with open(os.path.join(configuration['checkpoint_path'], 'class_distribution'), 'w') as w:
        w.write("Label distribution is:  " + str(freq))
    # Metrics
    test_losses = []
    train_losses = []

    global_accuracy_train = []
    global_gmean_train = []
    global_f1_train = []

    global_accuracy_test = []
    global_gmean_test = []
    global_f1_test = []

    epoch_log_interval = configuration["epoch_log_interval"]
    batch_log_interval = 10
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(starting_epoch, num_epochs + 1):

        print('Starting epoch {0}'.format(epoch))
        epoch_start_time = time.time()  # timer for entire epoch

        epoch_train_losses = []
        epoch_test_losses = []

        print('Starting training in epoch {0}'.format(epoch))
        model.train()
        for batch_idx, (data, labels, clusters) in enumerate(train_dataset_loader):

            optimizer.zero_grad()
            data = (data,)

            outputs = model(*data)

            outputs = (outputs,)
            targets = (labels,)
            clusters = (clusters.type(torch.IntTensor),)

            if configuration['weights_enabled'] == "True":
                criterion = BatchAllTripletWithClustersLossSemiHard(weights=train_weights, configuration=configuration)
            else:
                criterion = BatchAllTripletWithClustersLossSemiHard(weights=None, configuration=configuration)

            loss_outputs = criterion(*outputs, *targets, *clusters)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

            if (batch_idx + 1) % batch_log_interval == 0:
                print(
                    f"Training: {epoch}, {batch_idx + 1}\
                    Loss:{loss.item()}"
                )

        print('Starting testing in epoch {0}'.format(epoch))
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target, clusters) in enumerate(test_dataset_loader):
                data = (data,)

                outputs = model(*data)

                outputs = (outputs,)
                target = (target,)
                clusters = (clusters.type(torch.IntTensor),)

                if configuration['weights_enabled'] == "True":
                    criterion = BatchAllTripletWithClustersLossSemiHard(weights=test_weights,
                                                                        configuration=configuration)
                else:
                    criterion = BatchAllTripletWithClustersLossSemiHard(weights=None, configuration=configuration)

                loss_outputs = criterion(*outputs, *target, *clusters)

                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                epoch_test_losses.append(loss.item())

        # Updating global losses
        test_losses.append(np.mean(epoch_test_losses))
        train_losses.append(np.mean(epoch_train_losses))

        scheduler.step()

        # Saving model

        if epoch == num_epochs:
            print('Saving model at the end of epoch {0}'.format(epoch))
            path = os.path.join(configuration['checkpoint_path'], 'model.pth')
            torch.save(model.state_dict(), path)

        # Calculating metrics and creatign the embeddings

        if epoch % epoch_log_interval == 0:
            embeddings_train = []
            e_labels_train = []

            # train embeddings
            model.eval()
            with torch.no_grad():
                for data, target in embedding_train_loader:
                    embedding = model.embed(data)
                    embeddings_train.extend(embedding.numpy())
                    e_labels_train.extend(target.numpy())

            print("Creating chart of embedded samples and test KNN Classifier in epoch {0}".format(epoch))

            # vis.reduce_data_to_2d_with_pca(embeddings_train, e_labels_train,
            #                                os.path.join(configuration['checkpoint_path'], 'images'),
            #                                os.path.join(configuration['checkpoint_path'],
            #                                             'images/{}_train_epoch_{}.png'.format(
            #                                                 configuration['save_name'], epoch)), 'Reprezentacja ukryta')

            acc, gmean, f1, cfm, n_classes_ = test_knn_classifier(configuration, embeddings_train, e_labels_train)

            global_accuracy_train.append(acc)
            global_gmean_train.append(gmean)
            global_f1_train.append(f1)

            embeddings_test = []
            e_labels_test = []

            # test embeddings
            model.eval()
            with torch.no_grad():
                for data, target in embedding_test_loader:
                    embedding = model.embed(data)
                    embeddings_test.extend(embedding.numpy())
                    e_labels_test.extend(target.numpy())
            print("Creating chart of embedded samples and test KNN Classifier in epoch {0}".format(epoch))

            # vis.reduce_data_to_2d_with_pca(embeddings_test, e_labels_test,
            #                                os.path.join(configuration['checkpoint_path'], 'images'),
            #                                os.path.join(configuration['checkpoint_path'],
            #                                             'images/{}_test_epoch_{}.png'.format(
            #                                                 configuration['save_name'], epoch)), 'Embedding')

            acc, gmean, f1, cfm, n_classes_ = test_knn_classifier(configuration, embeddings_test, e_labels_test)

            global_accuracy_test.append(acc)
            global_gmean_test.append(gmean)
            global_f1_test.append(f1)

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time))

    plot_metrics(train_losses, os.path.join(configuration['checkpoint_path'],
                                            'images/{}_train_losses_plot.png'.format(configuration['save_name'])),
                 "train losses", os.path.join(configuration['checkpoint_path'],
                                              'images'))

    plot_metrics(test_losses, os.path.join(configuration['checkpoint_path'],
                                           'images/{}_test_losses_plot.png'.format(configuration['save_name'])),
                 "test_losses", os.path.join(configuration['checkpoint_path'],
                                             'images'))
    # plot_metrics(global_accuracy_train, os.path.join(configuration['checkpoint_path'],
    #                                                  'images/{}_accuracy_train_plot.png'.format(
    #                                                      configuration['save_name'])),
    #              "global_accuracy_train", os.path.join(configuration['checkpoint_path'],
    #                                                    'images'))
    plot_metrics(global_gmean_train, os.path.join(configuration['checkpoint_path'],
                                                  'images/{}_gmean_train_plot.png'.format(configuration['save_name'])),
                 "global_gmean_train", os.path.join(configuration['checkpoint_path'],
                                                    'images'))
    plot_metrics(global_f1_train, os.path.join(configuration['checkpoint_path'],
                                               'images/{}_f1_train_plot.png'.format(configuration['save_name'])),
                 "global_f1_train", os.path.join(configuration['checkpoint_path'],
                                                 'images'))

    # plot_metrics(global_accuracy_test, os.path.join(configuration['checkpoint_path'],
    #                                                 'images/{}_accuracy_test_plot.png'.format(
    #                                                     configuration['save_name'])),
    #              "global_accuracy_test", os.path.join(configuration['checkpoint_path'],
    #                                                   'images'))
    plot_metrics(global_gmean_test, os.path.join(configuration['checkpoint_path'],
                                                 'images/{}_gmean_test_plot.png'.format(configuration['save_name'])),
                 "global_gmean_test", os.path.join(configuration['checkpoint_path'],
                                                   'images'))
    plot_metrics(global_f1_test, os.path.join(configuration['checkpoint_path'],
                                              'images/{}_f1_test_plot.png'.format(configuration['save_name'])),
                 "global_f1_test", os.path.join(configuration['checkpoint_path'],
                                                'images'))

    isExist = os.path.exists(os.path.join(configuration['checkpoint_path'], 'metrics'))

    if not isExist:
        os.makedirs(os.path.join(configuration['checkpoint_path'], 'metrics'))
    #
    # np.save(os.path.join(configuration['checkpoint_path'],
    #                      'metrics/accuracy_train_{}.npy'.format(configuration['save_name']))
    #         , np.array(global_accuracy_train))
    # np.save(os.path.join(configuration['checkpoint_path'],
    #                      'metrics/gmean_train_{}.npy'.format(configuration['save_name']))
    #         , np.array(global_gmean_train))
    # np.save(os.path.join(configuration['checkpoint_path'],
    #                      'metrics/f1_train_{}.npy'.format(configuration['save_name']))
    #         , np.array(global_f1_train))
    #
    # np.save(os.path.join(configuration['checkpoint_path'],
    #                      'metrics/accuracy_test_{}.npy'.format(configuration['save_name']))
    #         , np.array(global_accuracy_test))
    # np.save(os.path.join(configuration['checkpoint_path'],
    #                      'metrics/gmean_test_{}.npy'.format(configuration['save_name']))
    #         , np.array(global_gmean_test))
    # np.save(os.path.join(configuration['checkpoint_path'],
    #                      'metrics/f1_test_{}.npy'.format(configuration['save_name']))
    #         , np.array(global_f1_test))

    np.save(os.path.join(configuration['checkpoint_path'],
                         'metrics/train_losses_{}.npy'.format(configuration['save_name']))
            , np.array(train_losses))
    np.save(os.path.join(configuration['checkpoint_path'],
                         'metrics/test_losses_{}.npy'.format(configuration['save_name']))
            , np.array(test_losses))

    # train embeddings
    final_embeddings = []
    final_labels = []

    model.eval()
    with torch.no_grad():
        for data, target in whole_dataset_loader:
            embedding = model.embed(data)
            final_embeddings.extend(embedding.numpy())
            final_labels.extend(target.numpy())

    vis.reduce_data_to_2d_with_pca(final_embeddings, final_labels,
                                   os.path.join(configuration['checkpoint_path'], 'images'),
                                   os.path.join(configuration['checkpoint_path'],
                                                'images/{}_whole_dataset_embedding.png'.format(
                                                    configuration['save_name'])), 'Reprezentacja Ukryta')

    knn_acc, knn_gmean, knn_f1, _, _ = stratifiedKCrossVKnn(configuration, final_embeddings,
                                                            final_labels)
    tree_acc, tree_gmean, tree_f1, _, _ = stratifiedKCrossVDecisionTree(configuration,
                                                                        final_embeddings,
                                                                        final_labels)
    _, _, _, knn_cfm, knn_n_classes_ = test_knn_classifier(configuration, final_embeddings,
                                                           final_labels)

    _, _, _, tree_cfm, tree_n_classes_ = test_tree_classifier(configuration,
                                                              final_embeddings,
                                                              final_labels)

    plot_conf_matrix(knn_cfm, knn_n_classes_, os.path.join(configuration['checkpoint_path'],
                                                           'images/knn_truth_table.png'),
                     os.path.join(configuration['checkpoint_path'],
                                  'images'))
    plot_conf_matrix(tree_cfm, tree_n_classes_, os.path.join(configuration['checkpoint_path'],
                                                             'images/tree_truth_table.png'),
                     os.path.join(configuration['checkpoint_path'],
                                  'images'))
    f = open(configuration['checkpoint_path'] + '/knn_metrics.txt', "w+")
    f.write(str(knn_acc) + "\n")
    f.write(str(knn_f1) + "\n")
    f.write(str(knn_gmean) + "\n")
    f.close()

    f = open(configuration['checkpoint_path'] + '/tree_metrics.txt', "w+")
    f.write(str(tree_acc) + "\n")
    f.write(str(tree_f1) + "\n")
    f.write(str(tree_gmean) + "\n")
    f.close()

import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter


def train(train_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level):
    task_criterion = nn.CrossEntropyLoss().cuda()
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        # Train model locally on source domains
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                     model_list[1:],
                                                                                     classifier_list[1:],
                                                                                     optimizer_list[1:],
                                                                                     classifier_optimizer_list[1:]):

            # check if the source domain is the malicious domain with poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            for i, (image_s, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break
                image_s = image_s.cuda()
                label_s = label_s.long().cuda()
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # each source domain do optimize
                feature_s = model(image_s)
                output_s = classifier(feature_s)
                task_loss_s = task_criterion(output_s, label_s)
                task_loss_s.backward()
                optimizer.step()
                classifier_optimizer.step()
    # Domain adaptation on target domain
    confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    # We use I(n_i>=1)/(N_T) to adjust the weight for knowledge distillation domain
    target_weight = [0, 0]
    consensus_focus_dict = {}
    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0
    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        if i >= batch_per_epoch:
            break
        optimizer_list[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()
        image_t = image_t.cuda()
        # Knowledge Vote
        with torch.no_grad():
            knowledge_list = [torch.softmax(classifier_list[i](model_list[i](image_t)), dim=1).unsqueeze(1) for
                              i in range(1, source_domain_num + 1)]
            knowledge_list = torch.cat(knowledge_list, 1)
        _, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate,
                                                                  num_classes=num_classes)
        target_weight[0] += torch.sum(consensus_weight).item()
        target_weight[1] += consensus_weight.size(0)
        # Perform data augmentation with mixup
        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
        feature_t = model_list[0](mixed_image)
        output_t = classifier_list[0](feature_t)
        output_t = torch.log_softmax(output_t, dim=1)
        task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))
        task_loss_t.backward()
        optimizer_list[0].step()
        classifier_optimizer_list[0].step()
        # Calculate consensus focus
        consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
                                                         source_domain_num, num_classes)
    # Consensus Focus Re-weighting
    target_parameter_alpha = target_weight[0] / target_weight[1]
    target_weight = round(target_parameter_alpha / (source_domain_num + 1), 4)
    epoch_domain_weight = []
    source_total_weight = 1 - target_weight
    for i in range(1, source_domain_num + 1):
        epoch_domain_weight.append(consensus_focus_dict[i])
    if sum(epoch_domain_weight) == 0:
        epoch_domain_weight = [v + 1e-3 for v in epoch_domain_weight]
    epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in
                           epoch_domain_weight]
    epoch_domain_weight.insert(0, target_weight)
    # Update domain weight with moving average
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)
    # Model aggregation and Batchnorm MMD
    federated_average(model_list, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # Recording domain weight in logs
    writer.add_scalar(tag="Train/target_domain_weight", scalar_value=target_weight, global_step=epoch + 1)
    for i in range(0, len(train_dloader_list) - 1):
        writer.add_scalar(tag="Train/source_domain_{}_weight".format(source_domains[i]),
                          scalar_value=domain_weight[i + 1], global_step=epoch + 1)
    print("Source Domains:{}, Domain Weight :{}".format(source_domains, domain_weight[1:]))
    return domain_weight


def test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes=126,
         top_5_accuracy=True):
    source_domain_losses = [AverageMeter() for i in source_domains]
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            output_t = classifier_list[0](model_list[0](image_t))
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        task_loss_t = task_criterion(output_t, label_t)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
    writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
                      global_step=epoch + 1)
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/target_domain_{}_accuracy_top1".format(target_domain).format(target_domain),
                      scalar_value=top_1_accuracy_t,
                      global_step=epoch + 1)
    if top_5_accuracy:
        top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/target_domain_{}_accuracy_top5".format(target_domain).format(target_domain),
                          scalar_value=top_5_accuracy_t,
                          global_step=epoch + 1)
        print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                          top_5_accuracy_t))
    else:
        print("Target Domain {} Accuracy {:.3f}".format(target_domain, top_1_accuracy_t))
    # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):
        tmp_score = []
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                output_s = classifier_list[s_i + 1](model_list[s_i + 1](image_s))
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = task_criterion(output_s, label_s)
            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
            tmp_score.append(torch.softmax(output_s, dim=1))
            # turn label into one-hot code
            tmp_label.append(label_onehot_s)
        writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        tmp_score = torch.cat(tmp_score, dim=0).detach()
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)
        _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/source_domain_{}_accuracy_top1".format(domain_s), scalar_value=top_1_accuracy_s,
                          global_step=epoch + 1)
        if top_5_accuracy:
            top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(tag="Test/source_domain_{}_accuracy_top5".format(domain_s), scalar_value=top_5_accuracy_s,
                              global_step=epoch + 1)

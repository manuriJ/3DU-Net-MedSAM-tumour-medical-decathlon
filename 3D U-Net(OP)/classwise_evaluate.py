import numpy as np
import tensorflow as tf


def dice_coefficient_per_class(y_true, y_pred, num_classes, smooth=1e-6):
    dice_scores = []
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class)
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    return tf.stack(dice_scores)


def iou_per_class(y_true, y_pred, num_classes, smooth=1e-6):
    iou_scores = []
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)
    return tf.stack(iou_scores)


def precision_per_class(y_true, y_pred, num_classes, smooth=1e-6):
    precision_scores = []
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        true_positives = tf.reduce_sum(y_true_class * y_pred_class)
        predicted_positives = tf.reduce_sum(y_pred_class)
        precision = (true_positives + smooth) / (predicted_positives + smooth)
        precision_scores.append(precision)
    return tf.stack(precision_scores)


def recall_per_class(y_true, y_pred, num_classes, smooth=1e-6):
    recall_scores = []
    for class_idx in range(num_classes):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        true_positives = tf.reduce_sum(y_true_class * y_pred_class)
        actual_positives = tf.reduce_sum(y_true_class)
        recall = (true_positives + smooth) / (actual_positives + smooth)
        recall_scores.append(recall)
    return tf.stack(recall_scores)


def calculate_class_metrics(model, val_generator, num_classes, class_weights):
    dice_scores = [[] for _ in range(num_classes)]
    iou_scores = [[] for _ in range(num_classes)]
    precision_scores = [[] for _ in range(num_classes)]
    recall_scores = [[] for _ in range(num_classes)]

    normalized_weights = np.array(class_weights) / np.sum(class_weights)

    for i in range(len(val_generator)):
        if val_generator.weighted_classes:
            x, y_true, _ = val_generator[i]
        else:
            x, y_true = val_generator[i]
        y_pred = model.predict(x)

        # Calculate metrics for each class per sample
        dice = dice_coefficient_per_class(y_true, y_pred, num_classes).numpy()
        iou = iou_per_class(y_true, y_pred, num_classes).numpy()
        precision = precision_per_class(y_true, y_pred, num_classes).numpy()
        recall = recall_per_class(y_true, y_pred, num_classes).numpy()

        for class_idx in range(num_classes):
            dice_scores[class_idx].append(dice[class_idx])
            iou_scores[class_idx].append(iou[class_idx])
            precision_scores[class_idx].append(precision[class_idx])
            recall_scores[class_idx].append(recall[class_idx])

    # Calculate weighted mean for each class
    mean_precision_per_class = [np.mean(scores) for i, scores in enumerate(precision_scores)]
    mean_recall_per_class = [np.mean(scores) for i, scores in enumerate(recall_scores)]
    mean_dice_per_class = [np.mean(scores) if scores else 0 for scores in dice_scores]
    mean_iou_per_class = [np.mean(scores) if scores else 0 for scores in iou_scores]

    # Calculate overall weighted means
    weighted_mean_dice = [mean * weight for mean, weight in zip(mean_dice_per_class, normalized_weights)]
    weighted_mean_iou = [mean * weight for mean, weight in zip(mean_iou_per_class, normalized_weights)]

    overall_mean_dice = np.sum(weighted_mean_dice)
    overall_mean_iou = np.sum(weighted_mean_iou)
    overall_mean_precision = np.mean(mean_precision_per_class)
    overall_mean_recall = np.mean(mean_recall_per_class)

    return (mean_dice_per_class, mean_iou_per_class, mean_precision_per_class, mean_recall_per_class,
            overall_mean_dice, overall_mean_iou, overall_mean_precision, overall_mean_recall)


def evaluate_segmentation(model, val_datagenerator, num_classes, sample_weights):
    mean_dice_per_class, mean_iou_per_class, mean_precision_per_class, mean_recall_per_class, overall_mean_dice, overall_mean_iou, overall_mean_precision, overall_mean_recall = calculate_class_metrics(
        model, val_datagenerator, num_classes, sample_weights)

    per_class = {'edma': {'DSC': mean_dice_per_class[1],
                          'IoU': mean_iou_per_class[1],
                          'Precision': mean_precision_per_class[1],
                          'Recall': mean_recall_per_class[1]},
                 'non': {'DSC': mean_dice_per_class[2],
                         'IoU': mean_iou_per_class[2],
                         'Precision': mean_precision_per_class[2],
                         'Recall': mean_recall_per_class[2]},
                 'enhancing': {'DSC': mean_dice_per_class[3],
                               'IoU': mean_iou_per_class[3],
                               'Precision': mean_precision_per_class[3],
                               'Recall': mean_recall_per_class[3]}
                 }

    results = {}
    results['Overall Mean Dice Coefficient'] = overall_mean_dice
    results['Overall Mean IoU'] = overall_mean_iou
    results['Overall Mean Precision'] = overall_mean_precision
    results['Overall Mean Recall'] = overall_mean_recall
    results['per_class'] = per_class

    return results


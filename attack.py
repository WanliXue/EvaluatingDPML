from classifier import train as train_model, get_predictions
from utilities import log_loss, prety_print_result, get_inference_threshold, generate_noise, get_random_features, get_attribute_variations, plot_sign_histogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
from scipy import stats
import numpy as np
import tensorflow as tf
import argparse
import os
import pickle

MODEL_PATH = './model/'
DATA_PATH = './data/'
RESULT_PATH = './results/'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_target_model(dataset=None, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn', privacy='no_privacy', dp='dp', epsilon=0.5, delta=1e-5, save=True):
    if dataset == None:
        dataset = load_data('target_data.npz', args)
    train_x, train_y, test_x, test_y = dataset

    classifier, aux = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio, silent=False, privacy=privacy, dp=dp, epsilon=epsilon, delta=delta)
    # test data for attack model
    attack_x, attack_y = [], []

    # data used in training, label is 1
    pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)

    attack_x.append(pred_scores)
    attack_y.append(np.ones(train_x.shape[0]))
    
    # data not used in training, label is 0
    pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)
    
    attack_x.append(pred_scores)
    attack_y.append(np.zeros(test_x.shape[0]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)

    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes, classifier, aux


def train_shadow_models(n_hidden=50, epochs=100, n_shadow=20, learning_rate=0.05, batch_size=100, l2_ratio=1e-7,
                        model='nn', save=True):
    attack_x, attack_y = [], []
    classes = []
    for i in range(n_shadow):
        #print('Training shadow model {}'.format(i))
        dataset = load_data('shadow{}_data.npz'.format(i), args)
        train_x, train_y, test_x, test_y = dataset

        # train model
        classifier = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                                   batch_size=batch_size, model=model, l2_ratio=l2_ratio)
        #print('Gather training data for attack model')
        attack_i_x, attack_i_y = [], []

        # data used in training, label is 1
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': train_x},
            num_epochs=1,
            shuffle=False)

        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_scores = get_predictions(predictions)
    
        attack_i_x.append(pred_scores)
        attack_i_y.append(np.ones(train_x.shape[0]))
    
        # data not used in training, label is 0
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': test_x},
            num_epochs=1,
            shuffle=False)

        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_scores = get_predictions(predictions)
    
        attack_i_x.append(pred_scores)
        attack_i_y.append(np.zeros(test_x.shape[0]))
        
        attack_x += attack_i_x
        attack_y += attack_i_y
        classes.append(np.concatenate([train_y, test_y]))
    # train data for attack model
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate(classes)

    if save:
        np.savez(MODEL_PATH + 'attack_train_data.npz', attack_x, attack_y)

    return attack_x, attack_y, classes


def train_attack_model(classes, dataset=None, n_hidden=50, learning_rate=0.01, batch_size=200, epochs=50,
                       model='nn', l2_ratio=1e-7):
    if dataset is None:
        dataset = load_attack_data()
    train_x, train_y, test_x, test_y = dataset

    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    true_y = []
    pred_y = []
    pred_scores = []
    true_x = []
    for c in unique_classes:
        #print('Training attack model for class {}...'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        classifier = train_model(c_dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio)
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': c_test_x},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        c_pred_y, c_pred_scores =  get_predictions(predictions)
        true_y.append(c_test_y)
        pred_y.append(c_pred_y)
        true_x.append(c_test_x)
        pred_scores.append(c_pred_scores)

    print('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    true_x = np.concatenate(true_x)
    pred_scores = np.concatenate(pred_scores)
    #print('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
    #print(classification_report(true_y, pred_y))
    prety_print_result(true_y, pred_y)
    fpr, tpr, thresholds = roc_curve(true_y, pred_y, pos_label=1)
    attack_adv = tpr[1] - fpr[1]
    return (attack_adv, pred_scores)


def save_data(args):
    print('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')

    target_size = args.target_data_size
    gamma = args.target_test_train_ratio

    x = pickle.load(open('dataset/'+args.train_dataset+'_features.p', 'rb'))
    y = pickle.load(open('dataset/'+args.train_dataset+'_labels.p', 'rb'))
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(x.shape, y.shape)

    # assert if data is enough for sampling target data
    assert(len(x) >= (1 + gamma) * target_size)
    x, train_x, y, train_y = train_test_split(x, y, test_size=target_size, stratify=y)
    print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
    x, test_x, y, test_y = train_test_split(x, y, test_size=gamma*target_size, stratify=y)
    print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))

    # save target data
    print('Saving data for target model')
    np.savez(DATA_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)

    # assert if remaining data is enough for sampling shadow data
    assert(len(x) >= (1 + gamma) * target_size)

    # save shadow data
    for i in range(args.n_shadow):
        print('Saving data for shadow model {}'.format(i))
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=target_size, test_size=gamma*target_size, stratify=y)
        print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
        print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))
        np.savez(DATA_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)


def load_data(data_name, args):
    target_size = args.target_data_size
    gamma = args.target_test_train_ratio
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]

    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)

    return train_x, train_y, test_x[:gamma*target_size], test_y[:gamma*target_size]


def shokri_membership_inference(args, attack_test_x, attack_test_y, test_classes):
    print('-' * 10 + 'SHOKRI\'S MEMBERSHIP INFERENCE' + '-' * 10 + '\n')    
    print('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n')
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        save=args.save_model)

    print('-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n')
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    return train_attack_model(
        dataset=dataset,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        n_hidden=args.attack_n_hidden,
        l2_ratio=args.attack_l2_ratio,
        model=args.attack_model,
        classes=(train_classes, test_classes))


def yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss=None):
    print('-' * 10 + 'YEOM\'S MEMBERSHIP INFERENCE' + '-' * 10 + '\n')    
    if test_loss == None:
    	pred_membership = np.where(per_instance_loss <= train_loss, 1, 0)
    else:
    	pred_membership = np.where(stats.norm(0, train_loss).pdf(per_instance_loss) >= stats.norm(0, test_loss).pdf(per_instance_loss), 1, 0)
    prety_print_result(membership, pred_membership)
    return pred_membership


def proposed_membership_inference(v_dataset, true_x, true_y, classifier, per_instance_loss, args):
    print('-' * 10 + 'PROPOSED MEMBERSHIP INFERENCE' + '-' * 10 + '\n')
    v_train_x, v_train_y, v_test_x, v_test_y = v_dataset
    v_true_x = np.vstack([v_train_x, v_test_x])
    v_true_y = np.concatenate([v_train_y, v_test_y])    
    v_pred_y, v_membership, v_test_classes, v_classifier, aux = train_target_model(
        dataset=v_dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        privacy=args.target_privacy,
        dp=args.target_dp,
        epsilon=args.target_epsilon,
        delta=args.target_delta,
        save=args.save_model)
    v_per_instance_loss = np.array(log_loss(v_true_y, v_pred_y))
    noise_params = (args.attack_noise_type, args.attack_noise_coverage, args.attack_noise_magnitude)
    v_counts = loss_increase_counts(v_true_x, v_true_y, v_classifier, v_per_instance_loss, noise_params)
    counts = loss_increase_counts(true_x, true_y, classifier, per_instance_loss, noise_params)
    return (v_membership, v_per_instance_loss, v_counts, counts)


def evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_thresholds):
    v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
    print('-' * 10 + 'Using Attack Method 1' + '-' * 10 + '\n')
    print('-' * 5 + 'Inference Maximizing Advantage' + '-' * 5 + '\n')
    thresh = get_inference_threshold(-v_per_instance_loss, v_membership)
    pred_membership = np.where(per_instance_loss <= -thresh, 1, 0)
    prety_print_result(membership, pred_membership)
    print('-' * 5 + 'Inference for fixed False Positive Rate' + '-' * 5 + '\n')
    for fpr_threshold in fpr_thresholds:
        print('FPR = %.2f' % fpr_threshold)
        thresh = get_inference_threshold(-v_per_instance_loss, v_membership, fpr_threshold)
        pred_membership = np.where(per_instance_loss <= -thresh, 1, 0)
        prety_print_result(membership, pred_membership)

    print('-' * 10 + 'Using Attack Method 2' + '-' * 10 + '\n')
    print('-' * 5 + 'Inference Maximizing Advantage' + '-' * 5 + '\n')
    thresh = get_inference_threshold(v_counts, v_membership)
    pred_membership = np.where(counts >= thresh, 1, 0)
    prety_print_result(membership, pred_membership)
    print('-' * 5 + 'Inference for fixed False Positive Rate' + '-' * 5 + '\n')
    for fpr_threshold in fpr_thresholds:
        print('FPR = %.2f' % fpr_threshold)
        thresh = get_inference_threshold(v_counts, v_membership, fpr_threshold)
        pred_membership = np.where(counts >= thresh, 1, 0)
        prety_print_result(membership, pred_membership)


def loss_increase_counts(true_x, true_y, classifier, per_instance_loss, noise_params, max_t=100):
    counts = np.zeros(len(true_x))
    for t in range(max_t):
        noisy_x = np.copy(true_x) + generate_noise(true_x.shape, true_x.dtype, noise_params)
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': noisy_x}, 
           num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_y = get_predictions(predictions)
        noisy_per_instance_loss = np.array(log_loss(true_y, pred_y))
        counts += np.where(noisy_per_instance_loss > per_instance_loss, 1, 0)
    return counts


def yeom_attribute_inference(true_x, true_y, classifier, membership, features, train_loss, test_loss=None):
    print('-' * 10 + 'YEOM\'S ATTRIBUTE INFERENCE' + '-' * 10 + '\n')
    pred_membership_all = []
    for feature in features:
        low_data, high_data, true_attribute_value = get_attribute_variations(true_x, feature)

        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': low_data},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, low_op = get_predictions(predictions)
        
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': high_data},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, high_op = get_predictions(predictions)

        low_op = low_op.astype('float32')
        high_op = high_op.astype('float32')
        low_op = log_loss(true_y, low_op)
        high_op = log_loss(true_y, high_op)

        high_prob = np.sum(true_attribute_value) / len(true_attribute_value)
        low_prob = 1 - high_prob

        if test_loss == None:
            pred_attribute_value = np.where(low_prob * stats.norm(0, train_loss).pdf(low_op) >= high_prob * stats.norm(0, train_loss).pdf(high_op), 0, 1)
            mask = [1]*len(pred_attribute_value)
        else:
            low_mem = np.where(stats.norm(0, train_loss).pdf(low_op) >= stats.norm(0, test_loss).pdf(low_op), 1, 0)
            high_mem = np.where(stats.norm(0, train_loss).pdf(high_op) >= stats.norm(0, test_loss).pdf(high_op), 1, 0)
            pred_attribute_value = [np.argmax([low_prob * a, high_prob * b]) for a, b in zip(low_mem, high_mem)]
            mask = [a | b for a, b in zip(low_mem, high_mem)]

        pred_membership = mask & (pred_attribute_value ^ true_attribute_value ^ [1]*len(pred_attribute_value))
        prety_print_result(membership, pred_membership)
        pred_membership_all.append(pred_membership)
    return pred_membership_all


def proposed_attribute_inference(true_x, true_y, classifier, membership, features, proposed_mi_outputs, args):
    print('-' * 10 + 'PROPOSED ATTRIBUTE INFERENCE' + '-' * 10 + '\n')
    v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
    low_per_instance_loss_all, high_per_instance_loss_all = [], []
    low_counts_all, high_counts_all = [], []
    true_attribute_value_all = []
    for feature in features:
        low_data, high_data, true_attribute_value = get_attribute_variations(true_x, feature)

        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': low_data},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, low_op = get_predictions(predictions)
        
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': high_data},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, high_op = get_predictions(predictions)

        low_op = low_op.astype('float32')
        high_op = high_op.astype('float32')
        low_op = log_loss(true_y, low_op)
        high_op = log_loss(true_y, high_op)
        
        noise_params = (args.attack_noise_type, args.attack_noise_coverage, args.attack_noise_magnitude)
        low_counts = loss_increase_counts(low_data, true_y, classifier, low_op, noise_params)
        high_counts = loss_increase_counts(high_data, true_y, classifier, high_op, noise_params)

        true_attribute_value_all.append(true_attribute_value)
        low_per_instance_loss_all.append(low_op)
        high_per_instance_loss_all.append(high_op)
        low_counts_all.append(low_counts)
        high_counts_all.append(high_counts)
    return (true_attribute_value_all, low_per_instance_loss_all, high_per_instance_loss_all, low_counts_all, high_counts_all)


def evaluate_proposed_attribute_inference(membership, proposed_mi_outputs, proposed_ai_outputs, features, fpr_thresholds):
    print('-' * 10 + 'Using Attack Method 1' + '-' * 10 + '\n')
    print('-' * 5 + 'Inference Maximizing Advantage' + '-' * 5 + '\n')
    evaluate_on_all_features(membership, proposed_mi_outputs, proposed_ai_outputs, features, fpr_threshold=None, attack_method=1)
    print('-' * 5 + 'Inference for fixed False Positive Rate' + '-' * 5 + '\n')
    for fpr_threshold in fpr_thresholds:
        print('FPR = %.2f' % fpr_threshold)
        evaluate_on_all_features(membership, proposed_mi_outputs, proposed_ai_outputs, features, fpr_threshold=fpr_threshold, attack_method=1)

    print('-' * 10 + 'Using Attack Method 2' + '-' * 10 + '\n')
    print('-' * 5 + 'Inference Maximizing Advantage' + '-' * 5 + '\n')
    evaluate_on_all_features(membership, proposed_mi_outputs, proposed_ai_outputs, features, fpr_threshold=None, attack_method=2)
    print('-' * 5 + 'Inference for fixed False Positive Rate' + '-' * 5 + '\n')
    for fpr_threshold in fpr_thresholds:
        print('FPR = %.2f' % fpr_threshold)
        evaluate_on_all_features(membership, proposed_mi_outputs, proposed_ai_outputs, features, fpr_threshold=fpr_threshold, attack_method=2)


def evaluate_on_all_features(membership, proposed_mi_outputs, proposed_ai_outputs, features, fpr_threshold=None, attack_method=1):
    v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
    true_attribute_value_all, low_per_instance_loss_all, high_per_instance_loss_all, low_counts_all, high_counts_all = proposed_ai_outputs
    for i in range(len(features)):
        high_prob = np.sum(true_attribute_value_all[i]) / len(true_attribute_value_all[i])
        low_prob = 1 - high_prob
        # Attack Method 1
        if attack_method == 1:
            thresh = get_inference_threshold(-v_per_instance_loss, v_membership, fpr_threshold)
            low_mem = np.where(low_per_instance_loss_all[i] <= -thresh, 1, 0)
            high_mem = np.where(high_per_instance_loss_all[i] <= -thresh, 1, 0)
        # Attack Method 2
        elif attack_method == 2:
            thresh = get_inference_threshold(v_counts, v_membership, fpr_threshold)
            low_mem = np.where(low_counts_all[i] >= thresh, 1, 0)
            high_mem = np.where(high_counts_all[i] >= thresh, 1, 0)
        pred_attribute_value = [np.argmax([low_prob * a, high_prob * b]) for a, b in zip(low_mem, high_mem)]
        mask = [a | b for a, b in zip(low_mem, high_mem)]
        pred_membership = mask & (pred_attribute_value ^ true_attribute_value_all[i] ^ [1]*len(pred_attribute_value))
        prety_print_result(membership, pred_membership)


def run_experiment(args):
    print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz', args)
    v_dataset = load_data('shadow0_data.npz', args)
    train_x, train_y, test_x, test_y = dataset
    true_x = np.vstack((train_x, test_x))
    true_y = np.append(train_y, test_y)
    batch_size = args.target_batch_size

    pred_y, membership, test_classes, classifier, aux = train_target_model(
        dataset=dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        privacy=args.target_privacy,
        dp=args.target_dp,
        epsilon=args.target_epsilon,
        delta=args.target_delta,
        save=args.save_model)
    train_loss, train_acc, test_loss, test_acc = aux
    per_instance_loss = np.array(log_loss(true_y, pred_y))
   
    features = get_random_features(true_x, range(true_x.shape[1]), 5)
    print(features)

    # Yeom's membership inference attack when only train_loss is known 
    yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
    # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
    yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)

    # Shokri's membership inference attack based on shadow model training
    #shokri_mem_adv, shokri_mem_confidence = shokri_membership_inference(args, pred_y, membership, test_classes)

    # Yeom's attribute inference attack when train_loss is known - Adversary 4 of Yeom et al.
    yeom_ai_outputs_1 = yeom_attribute_inference(true_x, true_y, classifier, membership, features, train_loss)
    # Yeom's attribute inference attack when both train_loss and test_loss are known - Adversary 7 of Yeom et al.
    yeom_ai_outputs_2 = yeom_attribute_inference(true_x, true_y, classifier, membership, features, train_loss, test_loss)

    fpr_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
    # Proposed membership inference attacks
    proposed_mi_outputs = proposed_membership_inference(v_dataset, true_x, true_y, classifier, per_instance_loss, args)
    evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_thresholds)

    # Proposed attribute inference attacks
    proposed_ai_outputs = proposed_attribute_inference(true_x, true_y, classifier, membership, features, proposed_mi_outputs, args)
    evaluate_proposed_attribute_inference(membership, proposed_mi_outputs, proposed_ai_outputs, features, fpr_thresholds)

    if not os.path.exists(RESULT_PATH+args.train_dataset+'_improved_mi'):
    	os.makedirs(RESULT_PATH+args.train_dataset+'_improved_mi')

    #pickle.dump([train_acc, test_acc, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, yeom_attr_mem, yeom_attr_pred, features], open(RESULT_PATH+args.train_dataset+'/'+args.target_model+'_'+args.target_privacy+'_'+args.target_dp+'_'+str(args.target_epsilon)+'_'+str(args.run)+'.p', 'wb'))
    if args.target_privacy == 'no_privacy':
        pickle.dump([aux, membership, per_instance_loss, features, yeom_mi_outputs_1, yeom_mi_outputs_2, yeom_ai_outputs_1, yeom_ai_outputs_2, proposed_mi_outputs, proposed_ai_outputs], open(RESULT_PATH+args.train_dataset+'_improved_mi/'+str(args.target_test_train_ratio)+args.target_model+'_'+args.target_privacy+'_'+str(args.target_l2_ratio)+'_'+str(args.run)+'.p', 'wb'))	
    else:
        pickle.dump([aux, membership, per_instance_loss, features, yeom_mi_outputs_1, yeom_mi_outputs_2, yeom_ai_outputs_1, yeom_ai_outputs_2, proposed_mi_outputs, proposed_ai_outputs], open(RESULT_PATH+args.train_dataset+'_improved_mi/'+str(args.target_test_train_ratio)+args.target_model+'_'+args.target_privacy+'_'+args.target_dp+'_'+str(args.target_epsilon)+'_'+str(args.run)+'.p', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--use_cpu', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))
    parser.add_argument('--target_test_train_ratio', type=int, default=1)
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='dp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)
    # proposed attack's noise parameters
    parser.add_argument('--attack_noise_type', type=str, default='gaussian')
    parser.add_argument('--attack_noise_coverage', type=str, default='full')
    parser.add_argument('--attack_noise_magnitude', type=float, default=0.01)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
    
    # Flag to disable GPU
    if args.use_cpu:
    	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.save_data:
        save_data(args)
    else:
        run_experiment(args)

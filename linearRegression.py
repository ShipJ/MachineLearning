import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import izip

data = np.loadtxt('spambase.data', delimiter=',')  # Load data into a numpy array
data_normalised = data.copy()

''' Normalise data '''
for i in range(data.shape[1]):  # Normalise data by subtracting feature means from each value
    feature = data[:, i]
    mean = np.mean(feature)  # Compute feature mean
    std = np.std(feature)  # Compute feature standard deviation
    feature = (feature - mean) / std  # Compute z-score based on computed mean and std
    data_normalised[:, i] = feature

k_folds = 10  # Number of cross-validation folds
feature_set = []
label_set = []
for i in range(k_folds):  # Split into 10 partitions as requested in assignment
    partition = data_normalised[i::k_folds]  # Select fold's based on requested indices
    np.random.shuffle(partition)  # Randomly shuffle each fold's samples
    feature_set.append(partition[:, :-1])
    label_set.append(partition[:, -1])

train_sets = []  # 10 training set feature sets
train_sets_labels = []  # 10 training set label sets
test_sets = []  # 10 test set feature sets
test_sets_labels = []  # 10 test set label sets
for i in range(k_folds):  # Construct training and test sets for each fold
    train_sets.append(np.array(np.concatenate(np.delete(feature_set, i))))  # Add all feature sets bar one for test set
    train_sets_labels.append(np.array(np.concatenate(np.delete(label_set, i))))  # Add corresponding training set labels
    test_sets.append(np.array(feature_set[i]))  # Add left-out set
    test_sets_labels.append(np.array(label_set[i]))  # Add corresponding test set labels

''' Linear/Logistic Regression Parameters '''
max_epochs = 1000  # Max number of epochs (Stochastic)
iterations = 1000  # Max number of iterations (Batch)
learn_rates = [0.1, 0.01, 0.001]  # Set the learning rate
tol = [0.0001, 0.00001, 0.0001, 0.00001]  # Convergence tolerance of each method

''' Linear Regression By Stochastic Gradient Descent'''
def stochastic_lin_reg(features, labels, rate, tol, epochs):
    [m, n] = features.shape  # Dimensions of feature set
    weight = np.zeros(n)  # Initialise weight as zero for all features
    count = 0
    initial_cost = 0
    updated_cost = 1
    cost_list = []  # Store cost for each sample
    for iteration in range(epochs):
        index = 0
        if abs(updated_cost - initial_cost) > tol:
            if updated_cost > 1000000:
                print 'Stochastic Linear Regression Divergence Warning: Learning Rate Too High'
                break
            else:
                total_cost = []
                for sample in features:  # For each sample in feature set, compute weight & cost
                    prediction = np.dot(np.transpose(weight), np.transpose(sample))  # Predict sample labels
                    error = prediction - labels[index]  # Error of predictions and true values
                    cost = (error ** 2) / (2 * m)  # Cost of all errors
                    total_cost.append(cost)
                    gradient = np.dot(np.transpose(sample), error) / m
                    weight -= (rate * gradient)  # Update the weight
                    index += 1
                initial_cost = updated_cost
                updated_cost = sum(total_cost)
                cost_list.append(updated_cost)
                count += 1
                if count == epochs - 1:
                    print 'Stochastic Linear Regression Maximised the Number of Epochs'
                    break
        else:
            print 'Stochastic Linear Regression Converged After %d Epochs' % count
            break
    return weight, cost_list, count


''' Linear Regression By Batch Gradient Descent'''
def batch_lin_reg(features, labels, rate, num_iterations, tol):
    [m, n] = features.shape  # Dimensions of feature set
    weight = np.zeros(n)  # Initialise weight as zeros
    cost_list = []  # List to store cost over each iteration
    initial_cost = 0
    updated_cost = 1
    count = 0
    for iteration in range(num_iterations):  # For each iteration, compute weight & cost
        if abs(updated_cost - initial_cost) > tol:
            if updated_cost > 10000000:
                print 'Batch Linear Regression Divergence warning: Learning rate too high'
                break
            else:
                prediction = np.dot(np.transpose(weight), np.transpose(features))  # Estimate sample labels
                error = prediction - labels  # Error of predictions and true values
                initial_cost = updated_cost
                cost = (error ** 2) / (2 * m)  # Cost of all errors
                updated_cost = sum(cost)
                gradient = (2 * np.dot(np.transpose(features), error)) / m
                weight -= (rate * gradient)  # Update the weight
                cost_list.append(sum(cost))  # Add cost to iteration list
                count += 1
                if count == num_iterations - 1:
                    print 'Batch Linear Regression Maximised the Number of Iterations'
                    break
        else:
            print 'Batch Linear Regression Converged After %d Iterations' % count
            break
    return weight, cost_list, count


''' Logistic Regression By Stochastic Gradient Descent'''
def stochastic_log_reg(features, labels, rate, tol, num_epochs):
    [m, n] = features.shape
    weight = np.zeros(n)
    cost_list = []
    initial_cost = 0
    updated_cost = 1
    count = 0
    for iteration in range(num_epochs):
        index = 0
        if abs(updated_cost - initial_cost) > tol:
            if updated_cost > 1000000:
                print 'Stochastic Logistic Regression Divergence Warning: Learning Rate Too High'
                break
            else:
                total_cost = []
                for sample in features:
                    prediction = np.divide(1, (1 + (math.e ** (-1 * np.dot(np.transpose(weight), np.transpose(sample))))))
                    error = prediction - labels[index]
                    if labels[index] > 0:
                        cost = -1 * np.log(prediction)
                    else:
                        cost = -1 * np.log(1-prediction)
                    total_cost.append(cost)
                    gradient = np.dot(np.transpose(sample), error) / m
                    weight -= (rate * gradient)
                    index += 1
                initial_cost = updated_cost
                updated_cost = sum(total_cost) / m
                cost_list.append(updated_cost)
                count += 1
                if count == num_epochs - 1:
                    print 'Stochastic Logistic Regression Maximised the Number of Epochs'
                    break
        else:
            print 'Stochastic Logistic Regression Converged After %d Epochs' % count
            break
    return weight, cost_list, count


''' Logistic Regression By Batch Gradient Descent'''
def batch_log_reg(features, labels, rate, num_iterations, tol):  # Batch Gradient Descent - Logistic Regression
    [m, n] = features.shape
    weight = np.zeros(n)
    cost_list = []
    initial_cost = 0
    updated_cost = 1
    count = 0
    for iteration in range(num_iterations):
        if abs(updated_cost - initial_cost) > tol:
            if updated_cost > 1000000:
                print 'Batch Logistic Regression Divergence Warning: Learning Rate Too High\n'
                break
            else:
                prediction = np.divide(1, (1 + (math.e ** (-1 * np.dot(np.transpose(weight), np.transpose(features))))))
                error = prediction - labels
                initial_cost = updated_cost
                cost = []
                for label in range(m):
                    if labels[label] > 0:
                        cost.append(-1 * np.log(prediction[i]))
                    else:
                        cost.append(-1 * np.log(1-prediction[i]))
                updated_cost = sum(cost) / m
                gradient = np.dot(np.transpose(features), error) / m
                weight -= (rate * gradient)
                cost_list.append(sum(cost))
                count += 1
                if count == num_iterations - 1:
                        print 'Batch Logistic Regression Maximised the Number of Iterations\n'
                        break
        else:
            print 'Batch Logistic Regression Converged After %d Iterations\n' % count
            break
    return weight, cost_list, count

for rate in learn_rates:
    predictions = [[] for i in range(4)]
    cost_history = [[] for i in range(4)]
    convergence_count = [[] for i in range(4)]
    for fold in range(k_folds):
        print 'Testing Fold %d: ' % (fold + 1)
        # Stochastic Linear Regression
        [stochastic_weight_linear, stochastic_cost_linear, stochastic_lin_count] = stochastic_lin_reg(train_sets[fold], train_sets_labels[fold], rate, tol[0], max_epochs)
        convergence_count[0].append(stochastic_lin_count)
        predictions[0].append(np.dot(test_sets[fold], stochastic_weight_linear))
        cost_history[0].append(stochastic_cost_linear)
        # Batch Linear Regression
        [batch_weight_linear, batch_cost_linear, batch_lin_count] = batch_lin_reg(train_sets[fold], train_sets_labels[fold], rate, iterations, tol[1])
        convergence_count[1].append(batch_lin_count)
        predictions[1].append(np.dot(test_sets[fold], batch_weight_linear))
        cost_history[1].append(batch_cost_linear)
        # # Stochastic Logistic Regression
        # [stochastic_weight_logistic, stochastic_cost_logistic, stochastic_log_count] = stochastic_log_reg(train_sets[fold], train_sets_labels[fold], rate, tol[2], max_epochs)
        # convergence_count[2].append(stochastic_log_count)
        # predictions[2].append(np.dot(test_sets[fold], stochastic_weight_logistic))
        # cost_history[2].append(stochastic_cost_logistic)
        # # Batch Logistic Regression
        # [batch_weight_logistic, batch_cost_logistic, batch_log_count] = batch_log_reg(train_sets[fold], train_sets_labels[fold], rate, iterations, tol[3])
        # convergence_count[3].append(batch_log_count)
        # predictions[3].append(np.dot(test_sets[fold], batch_weight_logistic))
        # cost_history[3].append(batch_cost_logistic)
    avg_stochastic_linear = [sum(x)/10.0 for x in izip(*cost_history[0])]
    avg_batch_linear = [sum(x)/10.0 for x in izip(*cost_history[1])]
    # avg_stochastic_logistic = [sum(x)/10.0 for x in izip(*cost_history[2])]
    # avg_batch_logistic = [sum(x)/10.0 for x in izip(*cost_history[3])]
    plt.plot(range(len(avg_stochastic_linear)), avg_stochastic_linear, label='Learning Rate: %f' % rate)
    plt.plot(range(len(avg_batch_linear)), avg_batch_linear, label='Learning Rate: %f' % rate)
    # plt.plot(range(len(avg_stochastic_logistic)), avg_stochastic_logistic, label='Learning Rate: %f' % rate)
    # plt.plot(range(len(avg_batch_logistic)), avg_batch_logistic, label='Learning Rate: %f' % rate)

plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.legend(loc='upper right')
plt.show()

''' True and False Positive Rates '''
def true_false_rate(predictions, true_labels, thresholds):
    m = len(predictions)
    tpr_threshold = []
    fpr_threshold = []
    threshold = min(predictions)
    for thresh in range(num_thresholds):
        true_false_pos_neg = [0 for k in range(4)]
        for index in range(m):
            if predictions[index] > threshold:
                if true_labels[index] > 0:
                    true_false_pos_neg[0] += 1
                else:
                    true_false_pos_neg[2] += 1
            else:
                if true_labels[index] <= 0:
                    true_false_pos_neg[1] += 1
                else:
                    true_false_pos_neg[3] += 1
        if true_false_pos_neg[0] != 0:
            tpr_threshold.append(1 - float(true_false_pos_neg[0]) / (true_false_pos_neg[0] + true_false_pos_neg[3]))
        else:
            tpr_threshold.append(0)
        if true_false_pos_neg[1] != 0:
            fpr_threshold.append(float(true_false_pos_neg[1]) / (true_false_pos_neg[1] + true_false_pos_neg[2]))
        else:
            fpr_threshold.append(0)
        threshold += (max(predictions) - min(predictions)) / thresholds
    return tpr_threshold, fpr_threshold

num_thresholds = 100  # Set number of thresholds
print 'Computing True and False Positive/Negative Rates...'
true_false_list = [[] for i in range(8)]  # Lists for true/false rates of each method
for fold in range(k_folds):  # For each fold, compute the true/false positive rates, for each method
    i = 0
    for j in range(4):  # Add rates for each method
        [tpr, fpr] = true_false_rate(predictions[j][fold], test_sets_labels[fold], num_thresholds)
        true_false_list[i].append(tpr)
        true_false_list[i+1].append(fpr)
        i += 2

x_y_values = [np.zeros(num_thresholds) for i in range(8)]
for fold in range(k_folds):
    for i in range(8):
        x_y_values[i] = np.add(x_y_values[i], np.array(true_false_list[i][fold]) / k_folds)

AUC_stochastic_linear = np.trapz(x_y_values[1], x_y_values[0], dx=num_thresholds)  # Stochastic AUC - Linear
print "Stochastic AUC (Linear) =", AUC_stochastic_linear
AUC_batch_linear = np.trapz(x_y_values[3], x_y_values[2], dx=num_thresholds)  # Batch AUC - Linear
print "Batch AUC (Linear) =", AUC_batch_linear
# AUC_stochastic_logistic = np.trapz(x_y_values[5], x_y_values[4], dx=num_thresholds)  # Stochastic AUC - Logistic
# print "Stochastic AUC (Logistic) =", AUC_stochastic_logistic
# AUC_batch_logistic = np.trapz(x_y_values[7], x_y_values[6], dx=num_thresholds)  # Batch AUC - Logistic
# print "Batch AUC (Logistic) =", AUC_batch_logistic

print 'Plotting ROC Curves...'  # Plot ROC curves for each regressor
plt.plot(x_y_values[0], x_y_values[1], label='Stochastic Linear')
plt.plot(x_y_values[2], x_y_values[3], label='Batch Linear')
# plt.plot(x_y_values[4], x_y_values[5], label='Stochastic Logistic')
# plt.plot(x_y_values[6], x_y_values[7], label='Batch Logistic')
plt.xlabel('Specificity (1 - False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate')
plt.legend(loc='lower right')
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--')
plt.show()

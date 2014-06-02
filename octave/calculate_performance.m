function [precision, recall, f_score, false_positives, false_negatives] = calculate_performance(theta, X_test, y_test)
%CALCULATE_PERFORMANCE Calculate performance of theta

predictions = svmpredict(y_test, X_test, theta);

false_positives = find(and((predictions == 1), (y_test == 0)));
false_negatives = find(and((predictions == 0), (y_test == 1)));


true_positives_count = sum(and((predictions == 1), (y_test == 1)))
true_negatives_count = sum(and((predictions == 0), (y_test == 0)))
false_positives_count = length(false_positives)
false_negatives_count = length(false_negatives)

true_positives_and_false_positives = true_positives_count + false_positives_count + 0.00001;
true_positives_and_false_negatives = true_positives_count + false_negatives_count + 0.00001;

precision = true_positives_count / true_positives_and_false_positives;
recall = true_positives_count / true_positives_and_false_negatives;

f_score = (2 * (precision * recall)) / (precision + recall + 0.00001);

end

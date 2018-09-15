% Classifying Hand-written Digits Using Gaussian Model
% Zephyr
% 02/20/2018
clear;
clc;
close all;
format long;
tic;
% Load MNIST dataset
trainData = loadMNISTImages('train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testData = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
trainImg = reshape(trainData, 28, 28, 60000);
testImg = reshape(testData, 28, 28, 10000);
trainData = trainData';
testData = testData';
% Extract 10000 data from train dataset to make valid dataset
validData = trainData(50001:end, :);
validLabels = trainLabels(50001:end);
vailidImg = trainImg(:, :, 50001:end);
trainData(50001:end, :) = [];
trainLabels(50001:end) = [];
trainImg(:, :, 50001:end) = [];
mu = zeros(10, 784);
sigma = zeros(784, 784, 10);
num = zeros(1, 10);
c = 0.02 : 0.001 : 0.1;
% Calculate mean value 'mu', covariance value 'sigma' and prior probability 'prior'
for a = 1 : 10
    mu(a, :) = mean(trainData(trainLabels==a-1, :));
    sigma(:, :, a) = cov(trainData(trainLabels==a-1, :));
    num(a) = sum(trainLabels==a-1);
end
priorP = num / 50000;
epochMax = numel(c);
% Training phase
validAccuracy = zeros(epochMax, 1);
estimatedValidLabels = zeros(10000, 1);
validConditionalP = zeros(10000, 10);
for epoch = 1 : epochMax
    disp(['Epoch: ' num2str(epoch) '/' num2str(numel(c)) ', c = ' num2str(c(epoch))]);
    for a = 1 : 10
        validConditionalP(:, a) = mvnpdf(validData, mu(a, :), sigma(:, :, a)+c(epoch)*eye(784));
    end
    count = 0;
% Estimated Labels = argmax(pierior*probability)
    validP = validConditionalP .* priorP;
    for a = 1 : 10000
        estimatedValidLabels(a) = find(validP(a, :)==max(validP(a, :)), 1) - 1;
        if estimatedValidLabels(a) == validLabels(a)
            count = count + 1;
        end
    end
    validAccuracy(epoch) = count / 10000;
    disp(['Valid data accuracy: ' num2str(validAccuracy(epoch)*100) '%']);
end
% Plot c-validAccuracy
figure;
plot(c, validAccuracy);
xlabel('c');
ylabel('Accuracy');
title('Valid Data Accuracy');
% Choose best c
cBest = c(find(validAccuracy == max(validAccuracy), 1));
% Test phase
testConditionalP = zeros(10000, 10);
for a = 1 : 10
    testConditionalP(:, a) = mvnpdf(testData, mu(a, :), sigma(:, :, a)+cBest*eye(784));
end
estimatedTestLabels = zeros(10000, 1);
count = 0;
for a = 1 : 10000
    testP = testConditionalP .* priorP;
    estimatedTestLabels(a) = find(testP(a, :)==max(testP(a, :)), 1) - 1;
    if estimatedTestLabels(a) == testLabels(a)
        count = count + 1;
    end
end
testAccuracy = count / 10000;
toc;
disp(['Test data accuracy: ' num2str(testAccuracy*100), '%']);
% Randomly choose 5 images from test dataset, estimate their lables and display.
d = ceil(10000 * rand(5, 1));
figure('NumberTitle', 'off', 'Name', 'Right Labels');
for a = 1 : 5
    subplot(1, 5, a);
    imshow(testImg(:, :, d(a)));
    title(num2str(testLabels(d(a))));
end
figure('NumberTitle', 'off', 'Name', 'Estimated Labels');
for a = 1 : 5
    subplot(1, 5, a);
    imshow(testImg(:, :, d(a)));
    title(num2str(estimatedTestLabels(d(a))));
end
clear all
close all

load KonIQ10k.mat

Constants.Directory = '/media/dvi/HD-B1/KonIQ-10k/1024x768';
Constants.numberOfImages = size(mos,1);
Constants.BaseCNN = 'inceptionv3';
Constants.numberOfTrainImages = round( 0.8*Constants.numberOfImages );
Constants.numberOfSplits = 20;
Constants.regressor = 'svr'; % svr, gpr, nn can be choosen
Constants.TransferLearning = false;

[net, Layers, length] = LoadBaseCNN(Constants);

Features = zeros(Constants.numberOfImages, length);

parfor i=1:Constants.numberOfImages
    if(mod(i,1000)==0)
        disp(i);
    end
    img           = imread( strcat(Constants.Directory, filesep, names{i}) );
    Features(i,:) = GetFeatures(img, net, Layers);
end

PLCC = zeros(1, Constants.numberOfSplits );
SROCC= zeros(1, Constants.numberOfSplits );
KROCC= zeros(1, Constants.numberOfSplits );

for i=1:Constants.numberOfSplits
    disp(i);
    p = randperm(numberOfImages);

    Target = mos(p);
    Data   = Features(p,:);
    PermutedNames = names(p);

    YTrain    = Target(1:Constants.numberOfTrainImages);
    DataTrain = Data(1:Constants.numberOfTrainImages,:);
    TrainNames= PermutedNames(1:Constants.numberOfTrainImages);

    YTest    = Target(Constants.numberOfTrainImages+1:end);
    DataTest = Data(Constants.numberOfTrainImages+1:end,:);
    TestNames= PermutedNames(Constants.numberOfTrainImages+1:end);

    if(strcmp(Constants.regressor,'svr'))
        Mdl  = fitrsvm(DataTrain,YTrain,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
    elseif(strcmp(Constants.regressor,'gpr'))
        Mdl  = fitrgp(DataTrain,YTrain,'KernelFunction','rationalquadratic','Standardize',true);
    else
        error('Undefined regressor type.');
    end
    YHat = predict(Mdl,DataTest);

    PLCC(i) =corr(YHat,YTest);
    SROCC(i)=corr(YHat,YTest,'Type','Spearman');
    KROCC(i)=corr(YHat,YTest,'Type','Kendall');
end

disp(round(mean(PLCC(:)),3)); disp(round(median(PLCC(:)),3)); disp(round(std(PLCC(:)),3));
disp(round(mean(SROCC(:)),3)); disp(round(median(SROCC(:)),3)); disp(round(std(SROCC(:)),3));

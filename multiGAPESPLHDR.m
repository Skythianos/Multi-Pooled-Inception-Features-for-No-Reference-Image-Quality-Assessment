clear all
close all

load ESPL_LIVE_HDR.mat  % This mat file contains information about ESPL LIVE HDR database

Constants.Directory = '/media/dvi/HD-B1/ESPL-LIVE-HDR/HDRDatabase/ESPL_LIVE_HDR_Database/Images'; % path to ESPL LIVE HDR database
Constants.numberOfImages = size(MOS,1); % number of images in ESPL LIVE HDR database
Constants.BaseCNN = 'inceptionv3';   % applied base convolutional neural network architecture
Constants.numberOfTrainImages = round( 0.8*Constants.numberOfImages );  % appx. 80% of images are used in training
Constants.numberOfSplits = 20;
Constants.regressor = 'gpr'; % svr or gpr can be choosen
Constants.TransferLearning = false;

[net, Layers, length] = LoadBaseCNN(Constants);

Features = zeros(Constants.numberOfImages, length);

PLCC = zeros(1, Constants.numberOfSplits );
SROCC= zeros(1, Constants.numberOfSplits );
KROCC= zeros(1, Constants.numberOfSplits );

mos  = MOS;
names= Names;

parfor i=1:Constants.numberOfImages
    if(mod(i,100)==0)
        disp(i);
    end
    img           = imread( strcat(Constants.Directory, filesep, names{i}) );
    Features(i,:) = GetFeatures(img, net, Layers);
end

for i=1:Constants.numberOfSplits
    disp(i);
    p = randperm(Constants.numberOfImages);

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

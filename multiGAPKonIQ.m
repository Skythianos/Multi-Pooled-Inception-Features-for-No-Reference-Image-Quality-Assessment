clear all
close all

load KonIQ10k.mat

Directory = '/media/dvi/HD-B1/KonIQ-10k/1024x768';
numberOfImages = size(mos,1);

BaseCNN = 'inceptionv3';

if(strcmp(BaseCNN,'googlenet'))
    net = googlenet;
    Layers = {'inception_3a-output', 'inception_3b-output', 'inception_4a-output',...
        'inception_4b-output', 'inception_4c-output', 'inception_4d-output',...
        'inception_4e-output', 'inception_5a-output', 'inception_5b-output'};
    %Layers = Layers(9);
    length = 5488;
    depth  = size(Layers,2);
elseif(strcmp(BaseCNN,'inceptionv3'))
    net = inceptionv3;
    Layers = {'mixed0','mixed1','mixed2','mixed3','mixed4','mixed5','mixed6',...
        'mixed7','mixed8','mixed9','mixed10'};
    %Layers = Layers(11);
    length = 10048;
    depth  = size(Layers,2);
elseif(strcmp(BaseCNN,'inceptionresnetv2'))
    net = inceptionresnetv2;
    Layers = {'mixed_5b','block35_1','block35_2','block35_3','block35_4','block35_5',...
        'block35_6','block35_7','block35_8','block35_9','block35_10','mixed_6a',...
        'block17_1','block17_2','block17_3','block17_4','block17_5','block17_6',...
        'block17_7','block17_8','block17_9','block17_10','block17_11','block17_12',...
        'block17_13','block17_14','block17_15','block17_16','block17_17','block17_18',...
        'block17_19','block17_20','mixed_7a','block8_1','block8_2','block8_3','block8_4',...
        'block8_5','block8_6','block8_7','block8_8','block8_9','block8_10'};
    length = size(Layers,2);
else
    error('Not supported base CNN');
end

Features = zeros(numberOfImages, length);

parfor i=1:numberOfImages
    if(mod(i,1000)==0)
        disp(i);
    end
    img           = imread( strcat(Directory, filesep, names{i}) );
    Features(i,:) = GetFeatures(img, net, Layers);
end

numberOfTrainImages = 8073;

for i=1:20
    disp(i);
    p = randperm(numberOfImages);

    Target = Target(p);
    Data   = Features(p,:);

    YTrain    = Target(1:numberOfTrainImages);
    DataTrain = Data(1:numberOfTrainImages,:);

    YTest    = Target(numberOfTrainImages+1:numberOfTrainImages+2000,:);
    DataTest = Data(numberOfTrainImages+1:numberOfTrainImages+2000,:);

    Mdl  = fitrsvm(DataTrain,YTrain,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
    YHat = predict(Mdl,DataTest);

    PLCC(i) =corr(YHat,YTest);
    SROCC(i)=corr(YHat,YTest,'Type','Spearman');
    KROCC(i)=corr(YHat,YTest,'Type','Kendall');
end

disp(round(mean(PLCC(:)),3)); disp(round(median(PLCC(:)),3)); disp(round(std(PLCC(:)),3));
disp(round(mean(SROCC(:)),3)); disp(round(median(SROCC(:)),3)); disp(round(std(SROCC(:)),3));
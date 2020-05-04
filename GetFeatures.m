function [Features] = GetFeatures(img, net, Layers)

    numberOfLayers = size(Layers,2);
    Features = [];
    
    for i=1:numberOfLayers
         featureMaps = activations(net, img, Layers{i}, 'OutputAs', 'channels');
         
         featureVector =  GlobalPooling(featureMaps, 'avg');       
         
         Features = [Features, featureVector];
    end

end
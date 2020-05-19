function [Features] = GetFeatures(img, net, Layers)

    if(isa(img,'uint8')==false)
        error('Variable img must be uint8 type');
    else
        if(ndims(img)~=3)
            if(ismatrix(img))
                tmp(:,:,1) = img;
                tmp(:,:,2) = img;
                tmp(:,:,3) = img;
                
                clear img
                img = tmp;
                clear tmp
            else
                error('Input image must be grayscale or RGB');
            end
        end
    end

    if(isa(net,'DAGNetwork')==false)
        error('Variable net must be DAGNetwork type');
    end

    numberOfLayers = size(Layers,2);
    Features = [];
    
    for i=1:numberOfLayers
         featureMaps = activations(net, img, Layers{i}, 'OutputAs', 'channels');
         
         featureVector =  GlobalPooling(featureMaps, 'avg');       
         
         Features = [Features, featureVector];
    end

end
function [g] = GlobalPooling(f, str)

    numberOfFeatureMaps = size(f,3);
    g = zeros(1,1,numberOfFeatureMaps);
    
    for i=1:numberOfFeatureMaps
        TMP = f(:,:,i);
        if(strcmp(str,'min'))
            g(i) = min(TMP(:));
        elseif(strcmp(str,'max'))
            g(i) = max(TMP(:));
        elseif(strcmp(str,'avg'))
            g(i) = mean(TMP(:));
        elseif(strcmp(str,'median'))
            g(i) = median(TMP(:));
        else
            error('Wrong input');
        end
    end
    
    g = reshape(g, [1 numberOfFeatureMaps]);

end
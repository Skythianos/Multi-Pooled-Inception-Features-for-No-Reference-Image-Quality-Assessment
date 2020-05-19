function [patches] = extractRandomPatches(img, num, sizeP)

    [H, W, ~] = size(img);
    
    for iterator=1:num
        r = randi(H-sizeP+1);
        c = randi(W-sizeP+1);
        
        patches{iterator} = img(r:r+sizeP-1, c:c+sizeP-1, :);
    end

end


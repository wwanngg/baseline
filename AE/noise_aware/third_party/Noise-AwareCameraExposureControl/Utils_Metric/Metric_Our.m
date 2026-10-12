
function [Value_Ours, Value_Grad, Value_Entropy, Value_Noise] = Metric_Our(Img_path)

Resize_factor   = 1.0;
Alpha            = 0.4;
Beta              = 0.4;

Kg = 2;
Lambda = 10^3;
Gamma = 0.06;

Ke = 0.125;
p = 0.10;
T_lower = 15;
T_upper = 235;

plot_flag = 0;

Img = imread(Img_path);
Img = imresize(Img,Resize_factor);
Value_Noise = zeros(size(Img,3),1);

for Channel = 1 : size(Img,3)
    Image  = uint8(Img(:,:,Channel));

    [H,W]   = size(Image);

    [Image_Grad, ]  = imgradient(double(Image), 'Sobel').^2;
    Grad_1D          = reshape(Image_Grad, [1,H*W]);
    Sort_Grad_1D   = sort(Grad_1D);
    Gth                = Sort_Grad_1D(1,int32(H*W*p));
    HomogenousRegionMask      = double(Image_Grad <= Gth);

    UnsaturatedMask = zeros(H,W);
    for i = 1 : H
        for j = 1: W
            if(T_lower <= Image(i,j) && Image(i,j) <= T_upper)
                UnsaturatedMask(i,j) = 1;
            else
                UnsaturatedMask(i,j) = 0;
            end
        end
    end

    UnSaturatedHomogenousMask = UnsaturatedMask .* HomogenousRegionMask;

    Ng = [ 1, -2,  1 ;
           -2,  4, -2 ;
            1, -2, 1  ];

    Laplacian_Image  = conv2(double(Image),double(Ng));
    Laplacian_Image  = Laplacian_Image(2:H+1,2:W+1);
    Masked_Laplacian_Image = Laplacian_Image.*UnSaturatedHomogenousMask;

    Ns                              = sum(sum(UnSaturatedHomogenousMask));
    Value_Noise(Channel)      = sqrt(pi/2) * (1/(6*Ns)) * sum(sum(abs(Masked_Laplacian_Image)));

    if(sum(sum(UnSaturatedHomogenousMask)) < H*W*0.0001)
        Value_Noise(Channel) = sqrt(pi/2)*(1/(6*(W-2)*(H-2)))*sum(sum(abs(Laplacian_Image)));
    end

    if plot_flag == 1
        figure;
        subplot(2,4,1);
        imagesc(Image);
        title('Input Img');

        subplot(2,4,2);
        imagesc(uint8(255*Image_Grad));
        title('Gradient Map');

        subplot(2,4,3);
        imagesc(uint8(255*HomogenousRegionMask));
        title('homogenous Mask Image');

        subplot(2,4,4);
        imshow(UnsaturatedMask);
        title('Unsaturated Mask Image');

        subplot(2,4,5);
        imshow(UnSaturatedHomogenousMask);
        title('UnSaturated Homogenous Mask Image');

        subplot(2,4,6);
        imagesc(uint8(Laplacian_Image));
        title('Laplacian image');

        subplot(2,4,7);
        imagesc(uint8(Masked_Laplacian_Image));
        title('Masked Laplacian image');
    end
end

if (size(Img_path,3) == 3)
    Value_Noise = (Value_Noise(1) + 2*Value_Noise(2) + Value_Noise(3)) / 4;
else
    Value_Noise = mean(Value_Noise);
end


gray_Image = rgb2gray(Img);
[H,W] = size(gray_Image);

[Image_Grad,] = imgradient(gray_Image);
G_norm = Image_Grad./sqrt(16 * 255 * 255 + 16 * 255 * 255);

Ng = log(Lambda.*(1-Gamma)+1);
Mapped_Grad = zeros(H,W);
for i=1:H
    for j=1:W
        if G_norm(i,j)>=Gamma
            Mapped_Grad(i,j)= log(Lambda * (G_norm(i,j) - Gamma) + 1)./ Ng;
        else
            Mapped_Grad(i,j) = 0;
        end
    end
end

num = 10;
Gridded_Grad = zeros(num,num);

for i = 1:num
    for j = 1:num
        Gridded_Grad(i,j) = mean2(Mapped_Grad(1+H/num * (i-1) : H/num * (i-1) +H/num,  1+W/num * (j-1) :W/num * (j-1) + W/num));
    end
end
Gridded_Grad_1D = reshape(Gridded_Grad, [num*num,1]);
Grad_mean = mean(Gridded_Grad_1D);
Std_num = std(Gridded_Grad_1D);
Value_Grad = Grad_mean / Std_num;

Value_Entropy = Ke*entropy(gray_Image);

Value_Ours = Alpha*Kg*Value_Grad + (1-Alpha)*Value_Entropy - Beta*Value_Noise;
end

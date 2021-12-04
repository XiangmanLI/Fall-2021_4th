clear
load a1digits.mat

% Conditional Gaussian Classifiers
C = size(digits_train,3); %class K
Size = size(digits_train,2); %matrix M
P = size(digits_train,1); %D features

Centre = mean(digits_train,2); 

uki = repmat(Centre,1,Size,1);
sigmasqr = sum(sum(sum((digits_train - uki).^2))) / (P .* Size .* C);
sigma = sqrt(sigmasqr);


figure
for i = 1:C
    subplot(5,2,i) 
    imagesc(reshape(Centre(:,1,i),8,8)'); 
    axis equal; 
    axis off; 
    colormap gray;   
    title([ '{\sigma} = ',num2str(sigma)])
end


%% Naive Bayes Classifiers
b = digits_train;
b(b >= 0.5)=1;
b(b < 0.5) = 0;
n = sum(b,2) / Size;
nre = reshape(n,P,C);

figure
for i = 1:C
    subplot(3,4,i)
    imagesc(reshape(nre(:,i),8,8)'); 
    axis equal; 
    axis off; 
    colormap gray;
    title(['Label   ',num2str(i)]);
end

%%Conditional Gaussian Classifiers
alpha = 0.1;
pCk = alpha;
class = 10;
data = 400;
pixel = 64; 

for z = 1:class
    for j = 1:data
        for i = 1:class
            exponent = exp(-1/(2*sigma)*sum((digits_test(:,j,z) - Centre(:,1,i)).^2,1));
            pxC(i) = (2*pi*sigma)^(-P/2).* exponent;
        end
        px = 1/400;
        pCx = pxC * pCk./ px;  
        [~,label(j)] = max(pCx); 
    end

    error(z) = 400-sum(label == z);
    ErrorR(z) = sum(label == z) / data;
end
OveralRate = sum(error)/4000 * 100;


fprintf('                ');
for i = 1:C
fprintf('       %d',i);
end
fprintf('\nGuass Classifier error:');
 
for i = 1:C
fprintf('      %d',error(i));
end
 
fprintf('\nGuass Classifier error(%%):');
 
for i = 1:C
fprintf('   %.2f',ErrorR(i)*100);
end

fprintf('\n\n OveralRate = ')
fprintf('   %.2f',OveralRate)

fprintf('\n')

%% Naive Bayes Classifiers
for z = 1:class
    for j = 1:data
        for i = 1:class
            bi = digits_test(:,j,z);
            bi(bi>=0.5)=1;
            bi(bi<0.5) = 0;
            ni = nre(:,i);
            pbcn(i) = 1;
            for x = 1:pixel
                if bi(x) ==1
                    pbcn(i) = pbcn(i) * ni(x);
                else
                    pbcn(i) = pbcn(i) * (1-ni(x));
                end
            end
        end
        pCkb = pbcn./sum(pbcn);
        [~,l(j)] = max(pCkb); 
    end
%%error and ErrorR
    errorB(z) = 400-sum(l == z);
    accB(z) = sum(l == z)/ data;
end

NOveralRate = sum(errorB)/4000 * 100;

fprintf('                ');
for i = 1:C
fprintf('       %d',i);
end
 
fprintf('\nBayes Classifier error:');
 
for i = 1:C
fprintf('      %d',errorB(i));
end
 
fprintf('\nBayes Classifier error(%%):');
 
for i = 1:C
fprintf('   %.2f',accB(i)*100);
end

fprintf('\n\n OveralRate = ')
fprintf('   %.2f',NOveralRate)

fprintf('\n')

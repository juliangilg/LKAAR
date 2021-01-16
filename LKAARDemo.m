% LKAAR DEMO.
% Julian Gil-Gonzalez.
% Universidad Tecnologica de Pereira.

clc; clear all; close all;

addpath(genpath('LKAAR'))

% First, we generate synthetic data from four 2-D Gaussian distributions 
% to simulate data from two classes (non-linear separable). 
K = 2; % number of classes
rng default  % For reproducibility
Nk = 15; % number of data for each Gaussian distribution
mu1 = [3 3]; mu2 = [3 -3]; mu3 = [-3 -3]; mu4 = [-3 3];
Cov = [0.8 0.1; 0.1 0.9];

X1 = mvnrnd(mu1,Cov,Nk); X3 = mvnrnd(mu3,Cov,Nk); % data for class 0
X2 = mvnrnd(mu2,Cov,Nk); X4 = mvnrnd(mu4,Cov,Nk); % data for class 1
X = [X1; X3; X2; X4];
y = [zeros(2*Nk,1); ones(2*Nk,1)];
 
% Plotting the data 
plot(X1(:,1), X1(:,2), 'rx','MarkerSize',5)
hold on
plot(X3(:,1), X3(:,2), 'rx','MarkerSize',5)
plot(X2(:,1), X2(:,2), 'bo','MarkerSize',5)
plot(X4(:,1), X4(:,2), 'bo','MarkerSize',5)

% Second, we compute a linear discriminant function for the gold
% standard based onn Gaussian processes (only for comparison purposes)


% configurate de GP model for classification
Ncg = 300; cov = {@covSEard}; sf = 1; ell = 0.7*ones(1, size(X,2));  
hyp0.cov  = log([ell,sf]); mean1 = {@meanZero}; hyp0.mean = [];
lik = 'likLogistic'; sn = 0.2; hyp0.lik = [];
inf = 'infLaplace';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = size(X,2);
% Generating the test data
x1 = -6:0.4:7; x2 = -5:0.4:6;
[x1,x2] = meshgrid(x1,x2);
x11 = x1(:); x22 = x2(:);
Xte = [x11 x22];

auxy = y; auxy(auxy==0) = -1;
hyp = minimize(hyp0,'gp', -Ncg, inf, mean1, cov, lik, X, auxy);
[~, ~, ~, ~, aux1] = gp(hyp, inf, mean1, cov, lik, X,...
                       auxy, Xte, ones(size(Xte,1), 1));
yast = exp(aux1); % prediction
yast = reshape(yast, size(x1)); % Reshape for visualization purposes
[C,h] = contour(x1,x2,yast, 'ShowText','on');
h.LineWidth = 1.5;
h.LevelStep = 0.1;
clabel(C,h,'FontSize',15)

% Third, we simulate annotations from 5 labelers (We follow the method 
% Biased coin (Non-homogeneous) )
R = 5;
Y = zeros(size(X,1), R);
[idxK, Cen] = kmeans(X, R);
pflip = [0.00,    0.90,   0.50,   0.15,  0.60;
         0.90,    0.00,   0.30,   0.40,  0.75;
         0.50,    0.30,   0.00,   0.60,  0.30;
         0.15,    0.40,   0.60,   0.00,  0.80;
         0.60,    0.75,   0.30,   0.80,  0.00];
for r = 1:R
    Y(:,r) = y;
    for k=1:R
        auxidx = find(idxK == k);
        Nki = length(auxidx);
        aux = binornd(1, pflip(k, r), Nki, 1);
        index = find(aux~=1);
        auxidx1 = auxidx;
        auxidx1(index) = [];
        Y(auxidx1,r) = ~y(auxidx1);
        Y(auxidx,r) = Y(auxidx,r) + 1;
    end
end

% Now, we compute the classification discriminant function using the annotations
% given for each labeler.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for r = 1:R
    auxY = Y(:,r);
    auxY(auxY == 1) = -1;
    auxY(auxY == 2) = 1;
    hyp = minimize(hyp0,'gp', -Ncg, inf, mean1, cov, lik, X, auxY);
    [~, ~, ~, ~, aux1] = gp(hyp, inf, mean1, cov, lik, X,...
                        auxY, Xte, ones(size(Xte,1), 1));
    yest(:,:,r) = [1-exp(aux1), exp(aux1)];
    yast = yest(:,2,r);
    
    yast = reshape(yast, size(x1));
    figure
    plot(X1(:,1), X1(:,2), 'rx','MarkerSize',5)
    hold on
    plot(X3(:,1), X3(:,2), 'rx','MarkerSize',5)
    plot(X2(:,1), X2(:,2), 'bo','MarkerSize',5)
    plot(X4(:,1), X4(:,2), 'bo','MarkerSize',5)
    [C,h] = contour(x1,x2,yast);
    h.LineWidth = 1.5;
    h.LevelStep = 0.1;
    clabel(C,h,'FontSize',15)
end

% Solution using our LKAAR

[Xaux, mux, stdx] = zscore(X);
beta = LKAAR(Xaux, Y);
beta = (beta).^2; 
beta = beta./repmat(sum(beta,2),1,R);
aux = Y==repmat(y+1,1,R);
disK = exp(-dist2(Xaux, Xte));
Nte = size(Xte,1);
muvec = zeros(Nte, K, R);
muvec1 = zeros(Nte, K, R);
for r = 1:R
   aux = (sum(disK.*repmat(beta(:,r),1,Nte))./sum(disK))'; 
   data = [Xte(:,1), Xte(:,2), aux];
   filename = ['Mu_Annota', num2str(r), '.dat'];
   save(filename, 'data', '-ascii')   % save to myfile.dat 
   muvec(:,:,r) = repmat(aux, 1, K);
end

aux1 = yest.*muvec;
aux1 = sum(aux1,3);
predtest = aux1(:,2);
predtest = reshape(predtest, size(x1));
figure
plot(X1(:,1), X1(:,2), 'rx','MarkerSize',5)
hold on
plot(X3(:,1), X3(:,2), 'rx','MarkerSize',5)
plot(X2(:,1), X2(:,2), 'bo','MarkerSize',5)
plot(X4(:,1), X4(:,2), 'bo','MarkerSize',5)
[C,h] = contour(x1,x2,predtest,'ShowText','on');
h.LineWidth = 1.5;
h.LevelStep = 0.1;
clabel(C,h,'FontSize',15)
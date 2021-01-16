function Q = LKAAR(X, Y)


R = size(Y, 2); % number of annotators
N = size(Y, 1);
KYYc = cell(R,1);
KYY = cell(R,1);
Q = zeros(N,R);
K = length(unique(Y));
if K ==1
    K = 2;
end

% computing the kernel over X
disK = dist2(X, X);
disK1 = tril(disK);
disK1(disK1 == 0) = NaN;
sig = 0.5*median(disK(:), 'omitNaN');
KXX = exp(-disK.^2/(sig));

% For the annotations, we use the 1-of-K codificaction 
N = size(X,1);
aux1 = cell(R,1);
for r = 1:R
    aux = zeros(N,K);
    auxy = zeros(N,K);
    for k = 1:K
        idx = find(Y(:,r) == k);
        aux(idx, k) = 1;
    end
    aux1{r} = aux;
end
Y1 = aux1;

% We center the kernel function
m = size(X,1); 
for r = 1:R
    KYY{r} = Y1{r}*Y1{r}';
    Kl{r} = 1.0*KXX + 0.0*KYY{r};
    Kl{r} = [ones(m,1) Kl{r}];
end

% We initialize the parameters using a centerd alignment solution
Ymv = MajorityVotKAAR(X, Y);
mu = Y == repmat(Ymv, 1, R);
auxmu = sum(mu,2);
auxmu(auxmu==0)=1;
mu = mu./repmat(auxmu, 1, R);
beta = mu;

% Now, we define the model parameters 
model.N = m; %number of instances
model.R = R; %number of annotators
model.KYY = KYY; % kernel over the annotations
model.KXX = KXX; % kernel over the features
model.KXXl = Kl; % kernel to estimate the weights q = KXXl*beta^m
model.beta = [ones(1, model.R); beta]; % initial parameters
model.betavec = model.beta(:);
model.sizep = (model.N+1)*model.R;

% We fix the options for optimization of parameters beta.
options = foptions;
options(1) = 1; 
options(2) = 1e-4; 
options(3) = 1e-4; 
options(9) = 1; 
options(14) = 300; 

thetaO = model.betavec;
% Now, it is necessary compute the parameters beta
model.betavec = scg('CostFunc', thetaO', options, 'GradFunc', model);
model.beta = reshape(model.betavec, (model.N+1), model.R);
for r = 1:R
   Q(:,r) = Kl{r}*model.beta(:,r);
end


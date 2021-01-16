function g = GradFunc(params, model)

if length(params)~= model.sizep
    error('The dimensions are not equal')
end

model.betavec = params;
model.beta = reshape(model.betavec, (model.N+1), model.R);
Q = cell(model.R,1);
Kmu = zeros(model.N);
q = model.KXXl{1}*model.beta;
for r = 1:model.R
    Q{r} = diag(q(:,r));
    Kmu = Kmu + Q{r}*model.KYY{r}*Q{r};
end

num = FrobeniusProduct(Kmu, model.KXX);
den = sqrt(FrobeniusProduct(Kmu, Kmu));
den2 = den^2;
dE_dQm = cell(model.R, 1);

for r = 1:model.R
    dE_dQm{r} = 2*(model.KYY{r}*Q{r}*model.KXX)*den;
    aux3 = zeros(model.N);
    
    for l = 1:model.R
        aux3 = aux3 + 4*model.KYY{r}*Q{r}*Q{l}*model.KYY{l}*Q{l};
    end
    
    dE_dQm{r} = (dE_dQm{r} - 1/2*den^(-1)*aux3*num)/den2;
end

dE_dbeta = zeros((model.N+1), model.R);

for r = 1:model.R
    aux = repmat(diag(dE_dQm{r}'), 1, model.N+1).*model.KXXl{r};
    dE_dbeta(:,r) = sum(aux);
end

g = -dE_dbeta(:)';
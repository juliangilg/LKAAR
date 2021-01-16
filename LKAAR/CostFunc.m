function E = CostFunc(params, model)

if length(params)~= model.sizep
    error('The dimensions are not equal')
end


model.betavec = params;
model.beta = reshape(model.betavec, (model.N+1), model.R);

Q = cell(model.R,1);
Kmu = zeros(model.N);
for r = 1:model.R
    q = model.KXXl{r}*model.beta(:,r);
    Q{r} = diag(q);
    Kmu = Kmu + Q{r}*model.KYY{r}*Q{r};
end
num = FrobeniusProduct(Kmu, model.KXX);
den = sqrt(FrobeniusProduct(Kmu, Kmu));
E = -num/den;
end
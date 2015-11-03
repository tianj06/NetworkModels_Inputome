% least square with constraints
% x = lsqlin(C,d,A,b)

load allUnitrawPSTH.mat

DAindex = strcmp(brainArea, 'Dopamine');
inputIndex = ~ismember(brainArea, {'Dopamine','VTA type3','VTA type2'});
striatumIndex = ismember(brainArea(inputIndex), {'Ventral striatum','Dorsal striatum'});

trialType = [1 2 4 5 6 7];
dataIndex = [];
for i = 1:length(trialType)
    dataIndex = [dataIndex (trialType(i)-1)*50+1:trialType(i)*50];
end

DA = mean(rawpsthAll(DAindex,dataIndex));
DA = normalize01(DA);
inputs = rawpsthAll(inputIndex,dataIndex);
for i = 1:size(inputs,1)
    inputs(i,:) = normalize01(inputs(i,:)); 
end

lb = -Inf(length(striatumIndex),1);
ub = double(striatumIndex');
ub(ub==0) = Inf;
ub(ub==1) = 0;
A = zeros(length(striatumIndex));
ind = find(striatumIndex);
for i = 1:length(ind)
    A(ind(i),ind(i)) = 1;
end
b = zeros(length(striatumIndex),1);

options = optimoptions('lsqlin','Algorithm','active-set');
x0 = 0.02*zeros(length(striatumIndex),1);
w = lsqlin(inputs',DA',A,b,[],[],lb,ub,x0, options);
%w = lsqlin(inputs',DA',A,b,[],[],lb,ub);

pred = inputs'*w;
figure;
subplot(2,1,1)
plot([pred DA'+0.5])
legend('prediction','DA')
subplot(2,1,2)
hist(w)





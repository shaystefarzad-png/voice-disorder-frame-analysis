function verifyLinearSVM
% Independent objective check before any real-data model selection or Test.
rng(1405,'twister');n=600;p=8;X=randn(n,p);Y=[zeros(200,1);ones(400,1)];
X(:,1:3)=X(:,1:3)+.7*Y;w=1+mod((1:n)',4);
reference=fitcsvm(X,Y,'KernelFunction','linear','BoxConstraint',1,'Standardize',false, ...
    'ClassNames',[0;1],'Prior','uniform','Weights',w,'Solver','SMO');
[fast,info]=fitclinear(X,Y,'Learner','svm','Regularization','ridge','Solver','dual', ...
    'Lambda',1/n,'IterationLimit',10000,'ClassNames',[0;1],'Prior','uniform','Weights',w);
nw=w;for c=0:1,nw(Y==c)=.5*w(Y==c)/sum(w(Y==c));end
margin=@(m)X*m.Beta+m.Bias;
objective=@(m).5*sum(m.Beta.^2)+n*sum(nw.*max(0,1-(2*Y-1).*margin(m)));
a=objective(reference);b=objective(fast);
assert(abs(a-b)/max(1,a)<.01,'Linear SVM objective mismatch.');
assert(mean(predict(reference,X)~=predict(fast,X))<.02,'Unexpected decision mismatch.');
assert(all(isfinite(fast.Beta)));
disp(struct('SMOObjective',a,'DualObjective',b,'TrainingDecisionAgreement',mean(predict(reference,X)==predict(fast,X))));
disp(info);
fprintf('LINEAR_SVM_OBJECTIVE_CHECK_PASSED\n');
end

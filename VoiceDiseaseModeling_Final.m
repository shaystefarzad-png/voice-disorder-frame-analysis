function VoiceDiseaseModeling_Final
% Independent implementation for MATLAB R2023b. No previous modeling code used.
% Run: VoiceDiseaseModeling_Final
% Requires Statistics and Machine Learning Toolbox; Parallel Computing Toolbox
% only when cfg.UseParallel=true. See deliverables/Implementation_Notes_FA.md.
% All reported metrics use SUBJECTS. CV estimates used for selection are not
% unbiased performance estimates; only the untouched test supports inference.

cfg = configuration();
rng(cfg.Seed,'twister');
assert(license('test','Statistics_Toolbox'), 'Statistics Toolbox is required.');
if cfg.QuickTest
    cfg.OutputDir = fullfile(cfg.OutputDir,'QUICK_TEST_NOT_FOR_INFERENCE');
    cfg.MaxFolds = 2; cfg.NumTrees = 8; cfg.NCAIterations = 10;
    cfg.MaxTrainFramesPerSubject = 9; cfg.Bootstrap = 100; cfg.Permutations = 100;
end
if ~isfolder(cfg.OutputDir), mkdir(cfg.OutputDir); end
if cfg.UseParallel
    assert(license('test','Distrib_Computing_Toolbox'), ...
        'Parallel Computing Toolbox required, or set UseParallel=false.');
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local',2);
    else
        assert(pool.NumWorkers==2,'Use an existing pool with exactly two workers.');
    end
end
names = featureNames();
writeRunProtocol(cfg,names);
healthy = loadPair(cfg,"Healthy",0,names);
results = cell(numel(cfg.Diseases),1);
runlog = table('Size',[0 3],'VariableTypes',{'string','string','string'}, ...
    'VariableNames',{'Disease','Status','Detail'});
for d = 1:numel(cfg.Diseases)
    disease = cfg.Diseases(d); started = tic;
    diseaseDir = fullfile(cfg.OutputDir,safeName(disease));
    if ~isfolder(diseaseDir), mkdir(diseaseDir); end
    files = [featurePath(cfg,25,"Healthy"),featurePath(cfg,75,"Healthy"), ...
        featurePath(cfg,25,disease),featurePath(cfg,75,disease)];
    if ~all(isfile(files))
        runlog(end+1,:) = {disease,"MissingInput",strjoin(files(~isfile(files))," | ")}; %#ok<AGROW>
        atomicTable(runlog,fullfile(cfg.OutputDir,'Run_Status.xlsx'),'Status');
        continue
    end
    sig = runSignature(cfg,files);
    cacheDir = fullfile(diseaseDir,['cache_' char(sig)]);
    if ~isfolder(cacheDir), mkdir(cacheDir); end
    completeFile = fullfile(cacheDir,'COMPLETE.mat');
    try
        if cfg.Resume && isfile(completeFile)
            z=load(completeFile,'result'); result=z.result;
            exportDisease(result,diseaseDir);
        else
            patient = loadPair(cfg,disease,1,names);
            result = runDisease(cfg,healthy,patient,disease,names,cacheDir);
            result.Signature=sig; result.Seconds=toc(started);
            exportDisease(result,diseaseDir);
            atomicSave(completeFile,'result',result);
        end
        results{d}=result;
        runlog(end+1,:)={disease,"Completed",sig}; %#ok<AGROW>
        exportSummary(results,cfg,runlog);
        statisticalReport(results,cfg);
        printResult(result,toc(started));
    catch err
        % Completed folds are individually persisted even if a worker dies.
        failure=struct('Disease',disease,'Identifier',err.identifier, ...
            'Report',getReport(err,'extended','hyperlinks','off'));
        atomicSave(fullfile(diseaseDir,'LastFailure.mat'),'failure',failure);
        runlog(end+1,:)={disease,"Failed",string(err.message)}; %#ok<AGROW>
        atomicTable(runlog,fullfile(cfg.OutputDir,'Run_Status.xlsx'),'Status');
    end
end
if ~any(~cellfun(@isempty,results))
    exportSummary(results,cfg,runlog);
else
    master=fullfile(cfg.OutputDir,'All_Diseases_Summary.xlsx');
    temporary=[tempname(cfg.OutputDir) '.xlsx']; copyfile(master,temporary);
    writetable(runlog,temporary,'Sheet','RunStatus','WriteMode','overwritesheet');
    movefile(temporary,master,'f');
end
end

function cfg=configuration()
cfg.InputDir = 'D:\SVD\meta\svd_wav\Features_Corrected';
cfg.OutputDir = 'D:\SVD\meta\svd_wav\ModelResults_Final';
% Edit disease membership ONLY here. Missing pairs are recorded, not invented.
cfg.Diseases = ["Rekurrensparese","Hyperfunktionelle","Laryngitis", ...
    "Reinke Ödem","Spasmodische Dysphonie","Stimmlippenpolyp"];
cfg.Seed=1405; cfg.UseParallel=true; cfg.QuickTest=false; cfg.Resume=true;
cfg.TestFraction=.20; cfg.MaxFolds=10;
cfg.Classifiers=["SVM","KNN","RandomForest","NaiveBayes"];
cfg.Rankers=["Ttest","CohenD","ReliefF","NCA"];
% Full mode uses every training frame. Finite values are an explicitly
% different, resource-limited experiment and enter the checkpoint signature.
cfg.MaxTrainFramesPerSubject=Inf;
cfg.NumTrees=40; cfg.MinLeafSize=5; cfg.KNNNeighbors=5;
cfg.SVMBackend='linear_dual'; % same linear hinge-loss model, scalable solver
cfg.NCAIterations=60; cfg.ReliefNeighbors=10;
cfg.CalibrationFraction=.20; % independent SUBJECT calibration for SVM
cfg.Bootstrap=5000; cfg.Permutations=10000; cfg.Alpha=.05;
cfg.TimeToleranceSec=.0051; cfg.EqualityTolerance=1e-10;
cfg.ProtocolVersion='independent-2026-11-04-v3-linear-dual';
end

function n=featureNames()
n.Time=["ZCR","ZCR_Var","RMS","PeakAmp","Entropy","EnergyEntropy", ...
    "JitterProxy_local","JitterProxy_rap","ShimmerProxy_local","ShimmerProxy_apq"];
n.Frequency=["HNR","SpecCentroid","SpecRolloff","SpecFlux","SpecEntropy", ...
    "MFCC"+string(1:13),"DeltaMFCC"+string(1:13),"FormantF1","FormantF2"];
n.All=[n.Time,n.Frequency];
assert(numel(n.All)==43 && numel(unique(n.All))==43);
end

function p=featurePath(cfg,ms,group)
p=string(fullfile(cfg.InputDir,sprintf('FullFeats_%dms_%s.xlsx',ms,group)));
end

function pair=loadPair(cfg,group,label,names)
a=readFeatureTable(featurePath(cfg,25,group),group,label,names);
b=readFeatureTable(featurePath(cfg,75,group),group,label,names);
[ia,ib,log]=matchFrames(a,b,group,cfg);
assert(~isempty(ia),'No matched frames in %s.',group);
pair.A=a; pair.B=b; pair.IA=ia; pair.IB=ib; pair.Log=log;
end

function t=readFeatureTable(path,group,label,names)
assert(isfile(path),'Missing feature workbook: %s',path);
opts=detectImportOptions(path,'VariableNamingRule','preserve');
required=["SubjectID","FileName","Vowel","Label","FrameID",names.All];
assert(all(ismember(required,string(opts.VariableNames))), ...
    'Workbook %s does not match the audited schema.',path);
opts=setvartype(opts,{'SubjectID','FileName','Vowel'},'string');
t=readtable(path,opts);
for col=["SubjectID","FileName","Vowel"]
    t.(col)=strip(string(t.(col)));
    assert(~any(ismissing(t.(col)) | strlength(t.(col))==0), ...
        'Missing %s in %s.',col,path);
end
% Never infer the subject from the filename; preserve the actual ID string.
assert(all(double(t.Label)==label),'Conflicting Label in %s.',path);
if ismember('Disease',t.Properties.VariableNames)
    assert(all(string(t.Disease)==group),'Conflicting Disease in %s.',path);
end
x=t{:,cellstr(names.All)};
assert(isnumeric(x),'Features must be numeric in %s.',path);
t{:,cellstr(names.All)}=double(x);
assert(all(isfinite(t.FrameID) & t.FrameID>=1 & t.FrameID==fix(t.FrameID)), ...
    'Invalid FrameID in %s.',path);
k=t(:,{'SubjectID','FileName','Vowel','FrameID'});
assert(height(unique(k,'rows'))==height(t),'Duplicate frame identity in %s.',path);
t=sortrows(t,{'SubjectID','FileName','Vowel','FrameID'});
end

function [ia,ib,log]=matchFrames(a,b,group,cfg)
ka=recordingKeys(a); kb=recordingKeys(b); keys=union(unique(ka),unique(kb));
ia=cell(numel(keys),1); ib=ia;
log=table('Size',[numel(keys) 10], ...
    'VariableTypes',{'string','string','string','string','double','double','double','string','double','double'}, ...
    'VariableNames',{'Disease','SubjectID','FileName','Vowel','Frames25','Frames75', ...
    'CommonFrames','Status','MeanTimeErrorSec','MaxTimeErrorSec'});
for j=1:numel(keys)
    i=find(ka==keys(j)); q=find(kb==keys(j));
    if ~isempty(i), row=a(i(1),:); else, row=b(q(1),:); end
    n=numel(i); m=numel(q); count=min(n,m); status="MatchedPosition";
    meanerr=NaN; maxerr=NaN;
    if n==0, status="Missing25";
    elseif m==0, status="Missing75";
    else
        hasCenter=ismember('FrameCenterTime',a.Properties.VariableNames) && ...
            ismember('FrameCenterTime',b.Properties.VariableNames);
        hasStart=ismember('FrameStartTime',a.Properties.VariableNames) && ...
            ismember('FrameStartTime',b.Properties.VariableNames);
        if hasCenter, time='FrameCenterTime'; elseif hasStart, time='FrameStartTime'; else, time=''; end
        if ~isempty(time)
            ta=double(a.(time)(i)); tb=double(b.(time)(q));
            assert(all(isfinite([ta;tb])) && all(diff(ta)>0) && all(diff(tb)>0), ...
                'Invalid/duplicated/nonmonotone frame times in %s.',row.FileName);
            if n>=m
                ix=nearestInjective(ta,tb); iy=(1:m)';
            else
                ix=(1:n)'; iy=nearestInjective(tb,ta);
            end
            errors=abs(ta(ix)-tb(iy)); meanerr=mean(errors); maxerr=max(errors);
            if maxerr>cfg.TimeToleranceSec
                % A shifted recording or an unsupported hop must not be
                % silently forced into an apparently valid hybrid pair.
                count=0; status="TimeMismatchExcluded";
            else
                ia{j}=i(ix); ib{j}=q(iy); status="Matched_"+string(time);
            end
        else
            ia{j}=i(uniformIndices(n,count)); ib{j}=q(uniformIndices(m,count));
        end
    end
    log(j,:)={group,row.SubjectID,row.FileName,row.Vowel,n,m,count,status,meanerr,maxerr};
end
ia=vertcat(ia{:}); ib=vertcat(ib{:});
assert(numel(ia)==numel(ib) && numel(unique(ia))==numel(ia) && numel(unique(ib))==numel(ib));
end

function ix=nearestInjective(long,short)
% Order-preserving nearest available time, reserving space for later frames.
% This is a one-to-one local nearest-time match, not an unconstrained join.
n=numel(long); m=numel(short); ix=zeros(m,1); previous=0; cursor=1;
for j=1:m
    lo=previous+1; hi=n-m+j; cursor=max(cursor,lo);
    while cursor<hi && abs(long(cursor+1)-short(j)) < abs(long(cursor)-short(j))-1e-12
        cursor=cursor+1;
    end
    ix(j)=cursor; previous=cursor;
end
end

function k=recordingKeys(t)
% Audited IDs contain no ASCII unit separator. Reject instead of colliding.
sep=string(char(31));
assert(~any(contains(t.SubjectID,sep)|contains(t.FileName,sep)|contains(t.Vowel,sep)));
k=t.SubjectID+sep+t.FileName+sep+t.Vowel;
end

function ix=uniformIndices(n,k)
if k==0, ix=zeros(0,1); elseif k==1, ix=ceil(n/2); else, ix=round(linspace(1,n,k))'; end
end

function result=runDisease(cfg,h,p,disease,names,cache)
assert(isempty(intersect(unique(h.A.SubjectID),unique(p.A.SubjectID))), ...
    'Healthy/patient SubjectID overlap: diagnosis/session resolution is required.');
% Primary comparison uses identical paired rows AND identical subjects for
% every scenario, eliminating sample-count and recording-set confounds.
a=[h.A(h.IA,:);p.A(p.IA,:)]; b=[h.B(h.IB,:);p.B(p.IB,:)];
assert(isequal(a.SubjectID,b.SubjectID) && isequal(a.FileName,b.FileName));
[ids,~,s]=unique(a.SubjectID,'sorted');
y=accumarray(s,double(a.Label),[],@mean);
assert(all(y==0|y==1),'A subject has conflicting labels.');
[train,test]=holdoutSubjects(y,cfg.TestFraction,cfg.Seed);
assert(~any(train & test) && all(train | test));
fold=zeros(numel(ids),1); fold(train)=groupFolds(y(train),cfg.MaxFolds,cfg.Seed+1);
cohort=table(ids,y,train,test,fold,'VariableNames',{'SubjectID','Label','Train','Test','Fold'});
excluded=excludedSubjects(h,p,ids);
atomicTable([h.Log;p.Log],fullfile(cache,'Matching.xlsx'),'Matching');
atomicTable(cohort,fullfile(cache,'SubjectSplit.xlsx'),'SubjectSplit');
atomicTable(excluded,fullfile(cache,'ExcludedSubjects.xlsx'),'Excluded');
X25=double(a{:,cellstr(names.All)}); X75=double(b{:,cellstr(names.All)});
D=cell(5,1);
D{1}=makeData(X25,s,y,ids,names.All);
D{2}=makeData(X75,s,y,ids,names.All);
D{3}=makeData([X25(:,1:10),X75(:,11:43)],s,y,ids, ...
    ["T25_"+names.Time,"F75_"+names.Frequency]);
D{4}=makeData(X25(:,1:10),s,y,ids,"T25_"+names.Time);
D{5}=makeData(X75(:,11:43),s,y,ids,"F75_"+names.Frequency);
labels=["S1_25ms","S2_75ms","S3A_Hybrid","Time25_Branch","Frequency75_Branch"];
searches=cell(5,1);
for j=1:5
    searches{j}=searchCV(D{j},find(train),fold(train),cfg, ...
        fullfile(cache,char(labels(j))),cfg.Seed+100*j);
end
% Nested outer CV: branch choice and hard-vote tie preference are learned
% exclusively inside each outer-training partition. Same outer folds as S1-S3A.
vote=selectVoting(D{4},D{5},find(train),fold(train),cfg,fullfile(cache,'Voting'));
vote.Best.K=searches{4}.Best.K+searches{5}.Best.K;
% Freeze baseline and hybrid choice BEFORE any test prediction is requested.
baseline=1;
if isBetter(searches{2}.Best,searches{1}.Best,cfg), baseline=2; end
hybrid=3;
if isBetter(vote.Best,searches{3}.Best,cfg), hybrid=4; end
selection=struct('Baseline',labels(baseline),'PreferredHybrid', ...
    string(choose(hybrid==3,'S3A_Hybrid','S3B_Voting')), ...
    'VotingMethod',vote.Method,'TimeChoice',searches{4}.Best, ...
    'FrequencyChoice',searches{5}.Best,'FrozenBeforeTest',true);
if isfile(fullfile(cache,'CI_SELECTION_FROZEN.mat'))
    frozen=load(fullfile(cache,'CI_SELECTION_FROZEN.mat'),'selection');
    assert(isequaln(selection,frozen.selection),'Selection differs from the pre-Test CI freeze.');
end
atomicSave(fullfile(cache,'TRAINING_SELECTION_FROZEN.mat'),'selection',selection);
% Train final models and evaluate each frozen scenario once. Test data never
% enters rankers, imputation, scaling, calibration, or selection.
pred=table(ids(test),y(test),'VariableNames',{'SubjectID','Label'});
models=cell(5,1); selected=cell(5,1); summary=table();
for j=1:5
    finalfile=fullfile(cache,sprintf('FinalModel_%d.mat',j));
    if cfg.Resume && isfile(finalfile)
        z=load(finalfile,'bundle'); bundle=z.bundle;
    else
        bundle=trainFinal(D{j},find(train),searches{j}.Best,cfg,cfg.Seed+10000+j);
        atomicSave(finalfile,'bundle',bundle);
    end
    models{j}=bundle; selected{j}=bundle.Features;
    ps=predictSubjects(bundle,D{j},find(test));
    if j<=3
        pred.(labels(j)+"_Probability")=ps; pred.(labels(j)+"_Prediction")=ps>=.5;
        summary=[summary;summaryRow(disease,labels(j),searches{j}.Best,y(test),ps>=.5)]; %#ok<AGROW>
    else
        pred.(labels(j)+"_Probability")=ps;
    end
end
pt=pred.Time25_Branch_Probability; pf=pred.Frequency75_Branch_Probability;
if vote.Method=="Soft"
    vp=(pt+pf)/2; vh=vp>=.5;
else
    preferTime=~isBetter(searches{5}.Best,searches{4}.Best,cfg);
    vh=hardVote(pt,pf,preferTime); vp=(pt+pf)/2;
end
pred.S3B_Voting_MeanProbability=(pt+pf)/2;
pred.S3B_Voting_Prediction=vh;
% In hard mode MeanProbability is descriptive, not the thresholded decision.
summary=[summary;summaryRow(disease,"S3B_Voting",vote.Best,y(test),vh)];
summary.Classifier(4)=searches{4}.Best.Classifier+" + "+searches{5}.Best.Classifier;
summary.Ranker(4)=searches{4}.Best.Ranker+" + "+searches{5}.Best.Ranker;
summary.FeatureCount(4)=searches{4}.Best.K+searches{5}.Best.K;
result=struct('Disease',disease,'Summary',summary,'Predictions',pred, ...
    'Cohort',cohort,'ExcludedSubjects',excluded,'Matching',[h.Log;p.Log], ...
    'Inventory',subjectInventory(h,p),'Selection',selection,'Searches',{searches}, ...
    'Voting',vote,'Features',{selected},'FinalRanks',{cellfun(@(b) ...
    table((1:numel(b.RankingOrder))',reshape(b.AllFeatureNames(b.RankingOrder),[],1), ...
    b.RankingScores(b.RankingOrder),'VariableNames',{'Rank','Feature','Score'}),models,'UniformOutput',false)}, ...
    'ScenarioNames',labels, ...
    'QuickTest',cfg.QuickTest,'TrainingFrameCap',cfg.MaxTrainFramesPerSubject);
end

function out=choose(condition,a,b)
if condition, out=a; else, out=b; end
end

function t=excludedSubjects(h,p,ids)
a=[h.A(:,{'SubjectID','Label'});h.B(:,{'SubjectID','Label'}); ...
    p.A(:,{'SubjectID','Label'});p.B(:,{'SubjectID','Label'})];
a=unique(a,'rows'); t=a(~ismember(a.SubjectID,ids),:);
t.Reason=repmat("No valid paired recording: excluded from ALL primary scenarios",height(t),1);
end

function t=subjectInventory(h,p)
t=table(); pairs={h,p};
for j=1:2
    for w=1:2
        if w==1, a=pairs{j}.A; ms=25; else, a=pairs{j}.B; ms=75; end
        [id,~,g]=unique(a.SubjectID);
        count=accumarray(g,1); nf=accumarray(g,(1:height(a))',[],@(ii)numel(unique(a.FileName(ii))));
        q=table(id,repmat(j-1,numel(id),1),repmat(ms,numel(id),1),nf,count, ...
            'VariableNames',{'SubjectID','Label','Window_ms','Files','Frames'});
        t=[t;q]; %#ok<AGROW>
    end
end
end

function d=makeData(x,s,y,ids,names)
d=struct('X',x,'S',s,'Y',y,'IDs',ids,'Names',names);
end

function [train,test]=holdoutSubjects(y,fraction,seed)
rng(seed,'twister'); train=true(numel(y),1); test=false(numel(y),1);
for c=0:1
    ix=find(y==c); n=numel(ix);
    assert(n>=5,'Need at least 5 subjects per class for holdout and nested voting.');
    ix=ix(randperm(n)); nt=max(1,min(n-4,round(fraction*n)));
    test(ix(1:nt))=true;
end
train=~test;
end

function f=groupFolds(y,maxfold,seed)
k=min([maxfold,sum(y==0),sum(y==1)]);
assert(k>=2,'Fewer than two training subjects in a class: CV is impossible.');
rng(seed,'twister'); f=zeros(numel(y),1);
for c=0:1
    ix=find(y==c); ix=ix(randperm(numel(ix)));
    f(ix)=mod((0:numel(ix)-1)',k)+1;
end
end

function search=searchCV(d,subjectIDs,fold,cfg,cache,seed)
if ~isfolder(cache), mkdir(cache); end
p=size(d.X,2); candidates=candidateTable(cfg,p); kfold=max(fold);
parts=cell(kfold,1); todo=false(kfold,1);
for k=1:kfold
    file=fullfile(cache,sprintf('Fold_%02d.mat',k));
    if cfg.Resume && isfile(file)
        z=load(file,'part'); parts{k}=z.part;
    else
        todo(k)=true;
    end
end
if cfg.UseParallel && any(todo)
    tasks=find(todo); newParts=cell(numel(tasks),1);
    parfor task=1:numel(tasks)
        k=tasks(task);
        part=evaluateFold(d,subjectIDs,fold,k,candidates,cfg,seed+k);
        atomicSave(fullfile(cache,sprintf('Fold_%02d.mat',k)),'part',part);
        newParts{task}=part;
    end
    parts(tasks)=newParts;
else
    for k=find(todo)'
        part=evaluateFold(d,subjectIDs,fold,k,candidates,cfg,seed+k);
        atomicSave(fullfile(cache,sprintf('Fold_%02d.mat',k)),'part',part);
        parts{k}=part;
    end
end
C=height(candidates); ba=NaN(kfold,C); se=ba; valid=true(1,C);
for k=1:kfold
    ba(k,:)=parts{k}.BA; se(k,:)=parts{k}.Sensitivity;
    valid=valid & isfinite(ba(k,:));
end
candidates.MeanBA=mean(ba,1)'; candidates.StdBA=std(ba,0,1)';
candidates.Sensitivity=mean(se,1)'; candidates.Valid=valid';
assert(any(valid),'All candidates failed. Inspect fold failure tables.');
if ~all(valid)
    failures=table();
    for k=1:kfold, failures=[failures;parts{k}.Failures]; end %#ok<AGROW>
    atomicTable(failures,fullfile(cache,'REQUIRED_CANDIDATE_FAILURES.xlsx'),'Failures');
    reasons=unique(string(failures.Error),'stable');
    error('Required candidates failed in %s. Causes: %s',cache,strjoin(reasons(1:min(8,numel(reasons)))," | "));
end
best=0;
for j=find(valid)
    b=table2struct(candidates(j,:));
    if best==0 || isBetter(b,table2struct(candidates(best,:)),cfg), best=j; end
end
search.Best=table2struct(candidates(best,:)); search.Candidates=candidates;
search.FoldBA=ba; search.FoldSensitivity=se;
search.Failures=table(); search.FoldRankings=table();
for k=1:kfold
    search.Failures=[search.Failures;parts{k}.Failures];
    search.FoldRankings=[search.FoldRankings;parts{k}.Rankings];
end
% No candidate test predictions are calculated or stored here.
end

function t=candidateTable(cfg,p)
counts=1:p;
if cfg.QuickTest, counts=unique([1,ceil(p/2),p]); end
[C,R,K]=ndgrid(1:numel(cfg.Classifiers),1:numel(cfg.Rankers),counts);
t=table(reshape(cfg.Classifiers(C(:)),[],1),reshape(cfg.Rankers(R(:)),[],1),K(:), ...
    'VariableNames',{'Classifier','Ranker','K'});
end

function part=evaluateFold(d,subjects,fold,k,candidates,cfg,seed)
training=subjects(fold~=k); validation=subjects(fold==k);
assert(isempty(intersect(training,validation)) && numel(unique(subjects))==numel(subjects));
% The same independent calibration partition is used by all SVM candidates.
[fitIDs,calIDs]=calibrationSplit(training,d.Y,cfg);
trainRows=trainingRows(d,training,cfg); valRows=find(ismember(d.S,validation));
pre=fitPreprocessor(d.X(trainRows,:));
z=applyPreprocessor(d.X(trainRows,:),pre);
zv=applyPreprocessor(d.X(valRows,:),pre);
svmPre=fitPreprocessor(d.X(trainRows(ismember(d.S(trainRows),fitIDs)),:));
zs=applyPreprocessor(d.X(trainRows,:),svmPre);
zsv=applyPreprocessor(d.X(valRows,:),svmPre);
ys=d.Y(d.S(trainRows)); w=subjectWeights(d.S(trainRows),ys);
fitMask=ismember(d.S(trainRows),fitIDs); calMask=ismember(d.S(trainRows),calIDs);
part.BA=NaN(1,height(candidates)); part.Sensitivity=part.BA;
part.Failures=table('Size',[0 5],'VariableTypes',{'double','string','string','double','string'}, ...
    'VariableNames',{'Fold','Classifier','Ranker','K','Error'});
part.Rankings=table();
for r=1:numel(cfg.Rankers)
    ranker=cfg.Rankers(r);
    try
        [order,score]=rankFeatures(z,d.S(trainRows),d.Y,ranker,cfg,seed+r);
    catch err
        part.Failures(end+1,:)={k,"ALL",ranker,0,string(err.message)};
        continue
    end
    rt=table(repmat(k,numel(order),1),repmat(ranker,numel(order),1), ...
        (1:numel(order))',reshape(d.Names(order),[],1),score(order), ...
        'VariableNames',{'Fold','Ranker','Rank','Feature','Score'});
    part.Rankings=[part.Rankings;rt]; %#ok<AGROW>
    for c=1:numel(cfg.Classifiers)
        activeOrder=order; activeX=z; activeV=zv;
        if cfg.Classifiers(c)=="SVM"
            try
                [activeOrder,svmScore]=rankFeatures(zs(fitMask,:),d.S(trainRows(fitMask)), ...
                    d.Y,ranker,cfg,seed+r);
                sr=table(repmat(k,numel(activeOrder),1),repmat(ranker+"_SVMFitOnly",numel(activeOrder),1), ...
                    (1:numel(activeOrder))',reshape(d.Names(activeOrder),[],1),svmScore(activeOrder), ...
                    'VariableNames',{'Fold','Ranker','Rank','Feature','Score'});
                part.Rankings=[part.Rankings;sr];
            catch err
                part.Failures(end+1,:)={k,"SVM",ranker,0,string(err.message)};
                continue
            end
            activeX=zs; activeV=zsv;
        end
        for n=1:numel(activeOrder)
            ci=find(candidates.Classifier==cfg.Classifiers(c)&candidates.Ranker==ranker&candidates.K==n);
            if isempty(ci), continue; end
            cols=activeOrder(1:n);
            try
                rng(seed+100*r+1000*c+n,'twister');
                model=fitModel(activeX(:,cols),ys,w,cfg.Classifiers(c), ...
                    fitMask,calMask,d.S(trainRows),cfg);
                fp=modelProbability(model,activeV(:,cols));
                pp=aggregateScores(fp,d.S(valRows),validation);
                m=metrics(d.Y(validation),pp>=.5);
                part.BA(ci)=m.BalancedAccuracy; part.Sensitivity(ci)=m.Sensitivity;
            catch err
                part.Failures(end+1,:)={k,cfg.Classifiers(c),ranker,n,string(err.message)};
            end
        end
    end
end
end

function [fitIDs,calIDs]=calibrationSplit(ids,y,cfg)
rng(cfg.Seed+50000,'twister'); calIDs=[];
for c=0:1
    q=ids(y(ids)==c); n=numel(q);
    assert(n>=2,'SVM calibration requires >=2 subjects per class in fitting partition.');
    q=q(randperm(n)); count=max(1,min(n-1,round(cfg.CalibrationFraction*n)));
    calIDs=[calIDs;q(1:count)]; %#ok<AGROW>
end
fitIDs=setdiff(ids,calIDs,'stable');
assert(isempty(intersect(fitIDs,calIDs)));
end

function rows=trainingRows(d,ids,cfg)
rows=find(ismember(d.S,ids));
if isfinite(cfg.MaxTrainFramesPerSubject)
    keep=cell(numel(ids),1);
    for j=1:numel(ids)
        q=rows(d.S(rows)==ids(j)); count=min(numel(q),cfg.MaxTrainFramesPerSubject);
        keep{j}=q(uniformIndices(numel(q),count));
    end
    rows=vertcat(keep{:});
end
end

function p=fitPreprocessor(x)
x(~isfinite(x))=NaN;
p.Median=median(x,1,'omitnan'); p.NoFinite=~isfinite(p.Median);
p.Median(p.NoFinite)=0;
x=fillMissing(x,p.Median);
p.Mu=mean(x,1); p.Sigma=std(x,0,1);
p.Sigma(~isfinite(p.Sigma)|p.Sigma<1e-12)=1;
end

function x=fillMissing(x,m)
for j=1:size(x,2), mask=~isfinite(x(:,j)); x(mask,j)=m(j); end
end

function z=applyPreprocessor(x,p)
z=(fillMissing(x,p.Median)-p.Mu)./p.Sigma;
assert(all(isfinite(z),'all'),'Nonfinite standardized values.');
end

function w=subjectWeights(s,y)
% Equal total weight per subject, then equal total weight per class.
[~,~,g]=unique(s); n=accumarray(g,1); w=1./n(g);
for c=0:1, mask=y==c; w(mask)=w(mask)/sum(w(mask)); end
w=w/sum(w)*numel(w);
end

function [order,score]=rankFeatures(z,s,y,method,cfg,seed)
% Rank on independent SUBJECT summaries, never treat correlated frames as
% independent observations for T tests or effect sizes. Fit models on frames.
[ids,~,g]=unique(s); u=zeros(numel(ids),size(z,2));
for j=1:size(z,2), u(:,j)=accumarray(g,z(:,j),[],@mean); end
y=y(ids); p=size(u,2); score=zeros(p,1);
rng(seed,'twister');
switch method
    case "Ttest"
        for j=1:p
            [~,pv,~,st]=ttest2(u(y==0,j),u(y==1,j),'Vartype','unequal');
            if isfinite(pv) && isfinite(st.tstat), score(j)=abs(st.tstat); end
            if isinf(st.tstat), score(j)=realmax; end
        end
    case "CohenD"
        n0=sum(y==0); n1=sum(y==1);
        pooled=sqrt(((n0-1)*var(u(y==0,:),0,1)+(n1-1)*var(u(y==1,:),0,1))/(n0+n1-2));
        delta=abs(mean(u(y==1,:),1)-mean(u(y==0,:),1));
        score=(delta./max(pooled,1e-12))';
    case "ReliefF"
        k=min([cfg.ReliefNeighbors,sum(y==0)-1,sum(y==1)-1]);
        assert(k>=1,'Not enough subjects for ReliefF.');
        [~,r]=relieff(u,y,k,'Method','classification','Prior','uniform'); score=r(:);
    case "NCA"
        nc=fscnca(u,y,'Standardize',false, ...
            'Lambda',.01,'Solver','lbfgs','IterationLimit',cfg.NCAIterations,'Verbose',0);
        score=nc.FeatureWeights(:);
    otherwise
        error('Unsupported ranker.');
end
score(~isfinite(score))=-realmax;
[~,order]=sortrows([-score,(1:p)'],[1 2]); order=order(:)';
end

function model=fitModel(x,y,w,kind,fitMask,calMask,subject,cfg)
options=statset('UseParallel',false);
model.Kind=kind;
switch kind
    case "SVM"
        % SVM raw margins are NOT probabilities. The margin model uses only
        % fit subjects; a Platt sigmoid uses separate calibration subjects.
        % Never call fitPosterior's frame-random cross-validation.
        if strcmp(cfg.SVMBackend,'linear_dual')
            % fitclinear minimizes weighted mean hinge loss + lambda*||b||^2/2.
            % lambda=1/(n*C) matches fitcsvm linear soft-margin C=1.
            wf=w(fitMask);yf=y(fitMask);
            for cls=0:1,wf(yf==cls)=.5*wf(yf==cls)/sum(wf(yf==cls));end
            [svm,fitInfo]=fitclinear(x(fitMask,:),yf,'Learner','svm', ...
                'Regularization','ridge','Solver','dual','Lambda',1/sum(fitMask), ...
                'IterationLimit',1e9,'PassLimit',1000,'DeltaGradientTolerance',1e-3, ...
                'ClassNames',[0;1],'Weights',wf);
            assert(all(fitInfo.TerminationCode>0),'Linear SVM did not converge.');
            if isfield(fitInfo,'Alpha'),fitInfo=rmfield(fitInfo,'Alpha');end
            model.SolverInfo=fitInfo;
        else
            svm=fitcsvm(x(fitMask,:),y(fitMask),'KernelFunction','linear', ...
                'BoxConstraint',1,'Standardize',false,'ClassNames',[0;1], ...
                'Prior','uniform','Weights',w(fitMask),'Solver','SMO');
        end
        [~,scores]=predict(svm,x(calMask,:)); raw=scores(:,svm.ClassNames==1);
        wc=subjectWeights(subject(calMask),y(calMask));
        model.Core=svm; model.Calibration=fitPlatt(raw,y(calMask),wc);
    case "KNN"
        model.Core=fitcknn(x,y,'NumNeighbors',min(cfg.KNNNeighbors,size(x,1)), ...
            'Distance','euclidean','DistanceWeight','inverse','Standardize',false, ...
            'ClassNames',[0;1],'Prior','uniform','Weights',w);
    case "RandomForest"
        tree=templateTree('MaxNumSplits',max(2,floor(size(x,1)/cfg.MinLeafSize)), ...
            'MinLeafSize',cfg.MinLeafSize,'NumVariablesToSample',max(1,floor(sqrt(size(x,2)))));
        model.Core=fitcensemble(x,y,'Method','Bag','NumLearningCycles',cfg.NumTrees, ...
            'Learners',tree,'ClassNames',[0;1],'Prior','uniform','Weights',w,'Options',options);
    case "NaiveBayes"
        % Weighted Gaussian NB with a predeclared variance floor handles
        % constant/all-missing-in-training columns without dropping features.
        for c=0:1
            mask=y==c; ww=w(mask)/sum(w(mask));
            mu=sum(x(mask,:).*ww,1); v=sum((x(mask,:)-mu).^2.*ww,1);
            model.Mu(c+1,:)=mu; model.Var(c+1,:)=max(v,1e-6);
        end
    otherwise
        error('Unsupported classifier.');
end
end

function b=fitPlatt(score,y,w)
% Ridge-regularized, class/subject-balanced Platt calibration, fitted only
% to held-out training subjects. Stable Newton iterations; no test labels.
scale=std(score); if scale<1e-12, scale=1; end
mu=mean(score); x=[ones(numel(score),1),(score-mu)/scale];
beta=zeros(2,1); penalty=diag([1e-8,1e-3]);
for j=1:60
    p=sigmoid(x*beta); g=x'*(w.*(p-y))+penalty*beta;
    H=x'*(x.*(w.*max(p.*(1-p),1e-8)))+penalty;
    step=H\g; old=plattLoss(x,beta,y,w,penalty); fraction=1;
    while fraction>1/1024
        proposal=beta-fraction*step;
        if plattLoss(x,proposal,y,w,penalty)<=old, break; end
        fraction=fraction/2;
    end
    beta=beta-fraction*step;
    if norm(fraction*step)<1e-7, break; end
end
b=struct('Beta',beta,'Mu',mu,'Scale',scale);
end

function value=plattLoss(x,b,y,w,penalty)
a=x*b; value=sum(w.*(max(a,0)+log1p(exp(-abs(a)))-y.*a))+.5*b'*penalty*b;
end

function p=sigmoid(z)
z=max(-35,min(35,z)); p=1./(1+exp(-z));
end

function p=modelProbability(model,x)
switch model.Kind
    case "SVM"
        [~,sc]=predict(model.Core,x); margin=sc(:,model.Core.ClassNames==1);
        b=model.Calibration; p=sigmoid([ones(size(margin)),(margin-b.Mu)/b.Scale]*b.Beta);
    case "NaiveBayes"
        logp=zeros(size(x,1),2);
        for c=1:2
            logp(:,c)=-.5*sum(log(2*pi*model.Var(c,:))+(x-model.Mu(c,:)).^2./model.Var(c,:),2);
        end
        p=sigmoid(logp(:,2)-logp(:,1));
    otherwise
        [~,sc]=predict(model.Core,x); p=sc(:,model.Core.ClassNames==1);
end
assert(all(isfinite(p)) && all(p>=-1e-8 & p<=1+1e-8),'Invalid probability.');
p=min(1,max(0,p));
end

function p=aggregateScores(frameScores,s,ids)
[tf,loc]=ismember(s,ids); assert(all(tf));
p=accumarray(loc,frameScores,[numel(ids),1],@mean,NaN);
assert(all(isfinite(p)),'A validation/test subject has no frame prediction.');
end

function bundle=trainFinal(d,ids,best,cfg,seed)
rows=trainingRows(d,ids,cfg);
[fitIDs,calIDs]=calibrationSplit(ids,d.Y,cfg);
rankMask=true(numel(rows),1);
if best.Classifier=="SVM", rankMask=ismember(d.S(rows),fitIDs); end
pre=fitPreprocessor(d.X(rows(rankMask),:));
z=applyPreprocessor(d.X(rows,:),pre);
[order,score]=rankFeatures(z(rankMask,:),d.S(rows(rankMask)),d.Y,best.Ranker,cfg,seed);
cols=order(1:best.K);
y=d.Y(d.S(rows)); w=subjectWeights(d.S(rows),y);
rng(seed,'twister');
model=fitModel(z(:,cols),y,w,best.Classifier,ismember(d.S(rows),fitIDs), ...
    ismember(d.S(rows),calIDs),d.S(rows),cfg);
bundle=struct('Preprocessor',pre,'Columns',cols,'Features',d.Names(cols), ...
    'RankingOrder',order,'RankingScores',score,'AllFeatureNames',d.Names, ...
    'Model',model,'Choice',best,'FitSubjectIDs',d.IDs(fitIDs), ...
    'CalibrationSubjectIDs',d.IDs(calIDs),'TrainingRows',numel(rows), ...
    'UsedTrainingSubjects',d.IDs(ids));
end

function p=predictSubjects(bundle,d,ids)
rows=find(ismember(d.S,ids)); z=applyPreprocessor(d.X(rows,:),bundle.Preprocessor);
p=aggregateScores(modelProbability(bundle.Model,z(:,bundle.Columns)),d.S(rows),ids);
end

function vote=selectVoting(time,freq,ids,fold,cfg,cache)
if ~isfolder(cache), mkdir(cache); end
kfold=max(fold); ptime=NaN(numel(ids),1); pfreq=ptime; hard=false(numel(ids),1);
ba=NaN(kfold,2); se=ba; details=table();
% Sequential outer loop. Each inner search parallelizes FOLDS only.
% No nested parfor, no additional pool, and no reuse of an outer validation
% label for branch selection or the hard-vote disagreement preference.
for k=1:kfold
    outerFile=fullfile(cache,sprintf('Outer_%02d.mat',k));
    if cfg.Resume && isfile(outerFile)
        z=load(outerFile,'part'); part=z.part;
    else
        tr=ids(fold~=k); va=ids(fold==k);
        inner=groupFolds(time.Y(tr),cfg.MaxFolds,cfg.Seed+20000+k);
        tc=searchCV(time,tr,inner,cfg,fullfile(cache,sprintf('Outer_%02d_Time',k)),cfg.Seed+30000+k);
        fc=searchCV(freq,tr,inner,cfg,fullfile(cache,sprintf('Outer_%02d_Frequency',k)),cfg.Seed+40000+k);
        tm=trainFinal(time,tr,tc.Best,cfg,cfg.Seed+50000+k);
        fm=trainFinal(freq,tr,fc.Best,cfg,cfg.Seed+60000+k);
        pt=predictSubjects(tm,time,va); pf=predictSubjects(fm,freq,va);
        prefer=~isBetter(fc.Best,tc.Best,cfg);
        part=struct('Time',pt,'Frequency',pf,'Hard',hardVote(pt,pf,prefer), ...
            'TimeChoice',tc.Best,'FrequencyChoice',fc.Best,'PreferTime',prefer);
        atomicSave(outerFile,'part',part);
    end
    mask=fold==k; ptime(mask)=part.Time; pfreq(mask)=part.Frequency; hard(mask)=part.Hard;
    m=metrics(time.Y(ids(mask)),part.Hard); ba(k,1)=m.BalancedAccuracy; se(k,1)=m.Sensitivity;
    m=metrics(time.Y(ids(mask)),(part.Time+part.Frequency)/2>=.5);
    ba(k,2)=m.BalancedAccuracy; se(k,2)=m.Sensitivity;
    q=table(k,string(part.TimeChoice.Classifier),string(part.TimeChoice.Ranker),part.TimeChoice.K, ...
        string(part.FrequencyChoice.Classifier),string(part.FrequencyChoice.Ranker), ...
        part.FrequencyChoice.K,part.PreferTime, ...
        'VariableNames',{'OuterFold','TimeClassifier','TimeRanker','TimeK', ...
        'FrequencyClassifier','FrequencyRanker','FrequencyK','PreferTime'});
    details=[details;q]; %#ok<AGROW>
end
m=[mean(ba,1)',std(ba,0,1)',mean(se,1)'];
choices=table(["Hard";"Soft"],m(:,1),m(:,2),m(:,3), ...
    'VariableNames',{'Method','MeanBA','StdBA','Sensitivity'});
a=struct('MeanBA',m(1,1),'StdBA',m(1,2),'Sensitivity',m(1,3),'K',0);
b=struct('MeanBA',m(2,1),'StdBA',m(2,2),'Sensitivity',m(2,3),'K',0);
winner=1; if isBetter(b,a,cfg), winner=2; end
vote.Method=choices.Method(winner); vote.Choices=choices;
vote.Best=struct('Classifier',"IndependentBranches",'Ranker',"IndependentBranches", ...
    'K',0,'MeanBA',m(winner,1),'StdBA',m(winner,2),'Sensitivity',m(winner,3));
vote.Details=details; vote.FoldBA=ba;
vote.TrainingOOF=table(time.IDs(ids),time.Y(ids),fold,ptime,pfreq,hard,(ptime+pfreq)/2>=.5, ...
    'VariableNames',{'SubjectID','Label','Fold','Ptime','Pfrequency','HardPrediction','SoftPrediction'});
end

function h=hardVote(pt,pf,preferTime)
a=pt>=.5; b=pf>=.5; h=a&b; disagreement=a~=b;
if preferTime, h(disagreement)=a(disagreement); else, h(disagreement)=b(disagreement); end
end

function yes=isBetter(a,b,cfg)
% Predeclared lexicographic tie-breaking. Stable enumeration resolves a
% complete tie; nothing here consults test labels or test predictions.
tol=cfg.EqualityTolerance;
if abs(a.MeanBA-b.MeanBA)>tol, yes=a.MeanBA>b.MeanBA; return; end
if abs(a.StdBA-b.StdBA)>tol, yes=a.StdBA<b.StdBA; return; end
if abs(a.Sensitivity-b.Sensitivity)>tol, yes=a.Sensitivity>b.Sensitivity; return; end
yes=a.K<b.K;
end

function m=metrics(y,h)
y=double(y(:)); h=logical(h(:));
m.TP=sum(y==1&h); m.TN=sum(y==0&~h); m.FP=sum(y==0&h); m.FN=sum(y==1&~h);
assert(m.TP+m.FN>0 && m.TN+m.FP>0,'Both classes are required.');
m.Sensitivity=m.TP/(m.TP+m.FN); m.Specificity=m.TN/(m.TN+m.FP);
m.BalancedAccuracy=(m.Sensitivity+m.Specificity)/2;
m.Accuracy=(m.TP+m.TN)/numel(y);
den=2*m.TP+m.FP+m.FN; if den==0, m.F1=0; else, m.F1=2*m.TP/den; end
end

function t=summaryRow(disease,scenario,best,y,h)
m=metrics(y,h);
t=table(disease,scenario,string(best.Classifier),string(best.Ranker),best.K, ...
    best.MeanBA,best.StdBA,m.Accuracy,m.BalancedAccuracy,m.Sensitivity, ...
    m.Specificity,m.F1,m.TP,m.TN,m.FP,m.FN,numel(y), ...
    'VariableNames',{'Disease','Scenario','Classifier','Ranker','FeatureCount', ...
    'CV_MeanBA','CV_StdBA','TestAccuracy','TestBalancedAccuracy','Sensitivity', ...
    'Specificity','F1','TP','TN','FP','FN','TestSubjects'});
end

function exportDisease(r,outdir)
book=fullfile(outdir,[safeName(r.Disease) '_Results.xlsx']);
tmp=[tempname(outdir) '.xlsx'];
writetable(r.Summary,tmp,'Sheet','Summary');
writetable(r.Predictions,tmp,'Sheet','SubjectPredictions');
writetable(r.Cohort,tmp,'Sheet','SubjectSplit');
writetable(r.ExcludedSubjects,tmp,'Sheet','ExcludedSubjects');
writetable(r.Inventory,tmp,'Sheet','SubjectInventory');
writetable(r.Matching,tmp,'Sheet','Matching');
sel=table(r.Selection.Baseline,r.Selection.PreferredHybrid,r.Selection.VotingMethod, ...
    r.Selection.FrozenBeforeTest,'VariableNames', ...
    {'BaselineChosenByTrainingCV','HybridChosenByTrainingCV','VotingMethod','FrozenBeforeTest'});
writetable(sel,tmp,'Sheet','FrozenSelections');
features=table();
for j=1:5
    name=r.ScenarioNames(j); s=r.Searches{j};
    writetable(s.Candidates,tmp,'Sheet',sprintf('CV_%d',j));
    writetable(s.Failures,tmp,'Sheet',sprintf('Failures_%d',j));
    writetable(s.FoldRankings,tmp,'Sheet',sprintf('FoldRankings_%d',j));
    writetable(r.FinalRanks{j},tmp,'Sheet',sprintf('FinalRank_%d',j));
    bf=table(repmat(name,numel(r.Features{j}),1),(1:numel(r.Features{j}))', ...
        reshape(r.Features{j},[],1),'VariableNames',{'Scenario','Rank','Feature'});
    features=[features;bf]; %#ok<AGROW>
    for ranker=unique(s.Candidates.Ranker,'stable')'
        drawCurve(s.Candidates,name,ranker,outdir);
    end
end
writetable(features,tmp,'Sheet','SelectedFeatures');
writetable(r.Voting.Choices,tmp,'Sheet','VotingSelection');
writetable(r.Voting.Details,tmp,'Sheet','VotingNestedBranches');
writetable(r.Voting.TrainingOOF,tmp,'Sheet','VotingTrainingOOF');
movefile(tmp,book,'f');
end

function drawCurve(t,scenario,ranker,outdir)
f=figure('Visible','off','Color','w'); cleaner=onCleanup(@()close(f)); %#ok<NASGU>
ax=axes(f); hold(ax,'on'); names=unique(t.Classifier,'stable');
colors=lines(numel(names));
for c=1:numel(names)
    q=t(t.Ranker==ranker & t.Classifier==names(c),:); q=sortrows(q,'K');
    errorbar(ax,q.K,100*q.MeanBA,100*q.StdBA,'Color',colors(c,:), ...
        'DisplayName',names(c),'LineWidth',1,'CapSize',2);
    valid=find(q.Valid);
    if ~isempty(valid)
        % Same scientific tie-break as the main selector, not simply max().
        best=valid(1);
        tol=1e-10;
        for j=valid(:)'
            a=q(j,:); b=q(best,:);
            if a.MeanBA>b.MeanBA+tol || ...
                    (abs(a.MeanBA-b.MeanBA)<=tol && a.StdBA<b.StdBA-tol) || ...
                    (abs(a.MeanBA-b.MeanBA)<=tol && abs(a.StdBA-b.StdBA)<=tol && ...
                    a.Sensitivity>b.Sensitivity+tol)
                best=j;
            end
        end
        plot(ax,q.K(best),100*q.MeanBA(best),'o','Color',colors(c,:), ...
            'MarkerFaceColor',colors(c,:),'HandleVisibility','off');
        text(ax,q.K(best),100*q.MeanBA(best),sprintf(' k=%d',q.K(best)), ...
            'Color',colors(c,:),'FontSize',8);
    end
end
xlabel(ax,'Number of features'); ylabel(ax,'Subject balanced accuracy (%)');
title(ax,scenario+" / "+ranker,'Interpreter','none');
grid(ax,'on'); legend(ax,'Location','best','Interpreter','none');
exportgraphics(f,fullfile(outdir,[safeName(scenario+"_"+ranker) '.png']),'Resolution',150);
end

function exportSummary(results,cfg,runlog)
t=table(); predictions=table(); features=table(); matching=table();
for j=1:numel(results)
    if isempty(results{j}), continue; end
    r=results{j}; t=[t;r.Summary]; %#ok<AGROW>
    p=r.Predictions; p=addvars(p,repmat(r.Disease,height(p),1),'Before',1,'NewVariableNames','Disease');
    predictions=[predictions;p]; matching=[matching;r.Matching]; %#ok<AGROW>
    for k=1:5
        q=table(repmat(r.Disease,numel(r.Features{k}),1), ...
            repmat(r.ScenarioNames(k),numel(r.Features{k}),1), ...
            (1:numel(r.Features{k}))',reshape(r.Features{k},[],1), ...
            'VariableNames',{'Disease','Scenario','Rank','Feature'});
        features=[features;q]; %#ok<AGROW>
    end
end
file=fullfile(cfg.OutputDir,'All_Diseases_Summary.xlsx'); tmp=[tempname(cfg.OutputDir) '.xlsx'];
if ~isempty(t)
    writetable(t,tmp,'Sheet','Summary'); writetable(predictions,tmp,'Sheet','SubjectPredictions');
    writetable(features,tmp,'Sheet','SelectedFeatures'); writetable(matching,tmp,'Sheet','Matching');
end
writetable(runlog,tmp,'Sheet','RunStatus');
metadata=table(string(cfg.ProtocolVersion),cfg.QuickTest,cfg.MaxTrainFramesPerSubject, ...
    numel(cfg.Diseases),sum(~cellfun(@isempty,results)), ...
    'VariableNames',{'Protocol','QuickTest','TrainingFrameCap','PlannedDiseases','CompletedDiseases'});
writetable(metadata,tmp,'Sheet','Protocol'); movefile(tmp,file,'f');
end

function statisticalReport(results,cfg)
% Two preregistered families: S3A vs baseline and S3B vs baseline, Holm across
% all PLANNED diseases within each. Two overall tests get a further Holm.
% If diseases are missing, the planned overall hypothesis is not confirmed.
rows=table(); planned=numel(cfg.Diseases); pairs=cell(planned,2);
for d=1:planned
    if isempty(results{d}), continue; end
    r=results{d}; t=r.Predictions; base=r.Selection.Baseline;
    basepred=double(t.(base+"_Prediction"));
    for h=1:2
        hybrid=string(choose(h==1,'S3A_Hybrid','S3B_Voting'));
        hp=double(t.(hybrid+"_Prediction"));
        [delta,lo,hi,pv]=pairedTest(t.Label,basepred,hp,cfg,cfg.Seed+70000+100*d+h);
        mb=metrics(t.Label,basepred); mh=metrics(t.Label,hp);
        q=table(r.Disease,hybrid,base,numel(t.Label),mb.BalancedAccuracy, ...
            mh.BalancedAccuracy,delta,lo,hi,pv,NaN,false, ...
            'VariableNames',{'Disease','Hybrid','Baseline','TestSubjects','BaselineBA', ...
            'HybridBA','DeltaBA','CI95_Lower','CI95_Upper','P_OneSided', ...
            'P_Holm','HypothesisSupported'});
        rows=[rows;q]; %#ok<AGROW>
        pairs{d,h}=table(t.SubjectID,t.Label,basepred,hp, ...
            'VariableNames',{'SubjectID','Label','BaselinePrediction','HybridPrediction'});
    end
end
if isempty(rows), return; end
for name=["S3A_Hybrid","S3B_Voting"]
    mask=rows.Hybrid==name;
    rows.P_Holm(mask)=holm(rows.P_OneSided(mask),planned);
end
rows.HypothesisSupported=rows.DeltaBA>0 & rows.CI95_Lower>0 & rows.P_Holm<cfg.Alpha & ~cfg.QuickTest;
globalrows=table();
for h=1:2
    subset=pairs(:,h); subset=subset(~cellfun(@isempty,subset));
    [delta,lo,hi,pv]=globalPairedTest(subset,cfg,cfg.Seed+80000+h);
    complete=numel(subset)==planned;
    q=table(string(choose(h==1,'S3A_Hybrid','S3B_Voting')),numel(subset),planned, ...
        delta,lo,hi,pv,NaN,complete,false, ...
        'VariableNames',{'Hybrid','IncludedDiseases','PlannedDiseases','MeanDeltaBA', ...
        'CI95_Lower','CI95_Upper','P_OneSided','P_Holm','AllPlannedDiseasesAvailable','HypothesisSupported'});
    globalrows=[globalrows;q]; %#ok<AGROW>
end
globalrows.P_Holm=holm(globalrows.P_OneSided,2);
globalrows.HypothesisSupported=globalrows.MeanDeltaBA>0 & globalrows.CI95_Lower>0 & ...
    globalrows.P_Holm<cfg.Alpha & globalrows.AllPlannedDiseasesAvailable & ~cfg.QuickTest;
comparison=table();
for d=1:planned
    if isempty(results{d}), continue; end
    r=results{d}; a=rows(rows.Disease==r.Disease & rows.Hybrid=="S3A_Hybrid",:);
    b=rows(rows.Disease==r.Disease & rows.Hybrid=="S3B_Voting",:);
    q=table(r.Disease,r.Selection.Baseline,r.Selection.PreferredHybrid, ...
        a.DeltaBA,b.DeltaBA,a.DeltaBA>0,b.DeltaBA>0, ...
        a.HypothesisSupported,b.HypothesisSupported, ...
        'VariableNames',{'Disease','TrainingSelectedBaseline','TrainingPreferredHybrid', ...
        'S3A_DeltaBA','S3B_DeltaBA','S3A_TestBAHigher','S3B_TestBAHigher', ...
        'S3A_HypothesisSupported','S3B_HypothesisSupported'});
    comparison=[comparison;q]; %#ok<AGROW>
    book=fullfile(cfg.OutputDir,safeName(r.Disease),[safeName(r.Disease) '_Results.xlsx']);
    temporary=[tempname(fileparts(book)) '.xlsx']; copyfile(book,temporary);
    writetable([a;b],temporary,'Sheet','StatisticalHypothesis','WriteMode','overwritesheet');
    movefile(temporary,book,'f');
end
file=fullfile(cfg.OutputDir,'Statistical_Hypothesis_Validation.xlsx');
tmp=[tempname(cfg.OutputDir) '.xlsx'];
writetable(rows,tmp,'Sheet','PerDisease'); writetable(globalrows,tmp,'Sheet','Overall');
writetable(comparison,tmp,'Sheet','Comparison');
notes=table(["DeltaBA is a fraction; multiply by 100 for percentage points."; ...
    "95% CIs: paired, class-stratified subject bootstrap percentile intervals."; ...
    "P-values: paired one-sided permutation, alternative hybrid BA > baseline BA."; ...
    "Holm per hybrid family includes every planned disease, even missing tests."; ...
    "Overall resampling/swap indicators synchronize shared SubjectIDs across diseases."; ...
    "Overall tests are Holm adjusted across the two hybrid methods."; ...
    "Incomplete disease sets and QuickTest cannot confirm the planned overall hypothesis."; ...
    "Significance is conditional on this split and trained models; it is not external validation."], ...
    'VariableNames',{'Interpretation'});
writetable(notes,tmp,'Sheet','Methods'); movefile(tmp,file,'f');
master=fullfile(cfg.OutputDir,'All_Diseases_Summary.xlsx');
temporary=[tempname(cfg.OutputDir) '.xlsx']; copyfile(master,temporary);
writetable(rows,temporary,'Sheet','StatisticalHypothesis','WriteMode','overwritesheet');
writetable(globalrows,temporary,'Sheet','OverallHypothesis','WriteMode','overwritesheet');
writetable(comparison,temporary,'Sheet','Comparison','WriteMode','overwritesheet');
movefile(temporary,master,'f');
end

function [delta,lo,hi,pvalue]=pairedTest(y,b,h,cfg,seed)
rng(seed,'twister'); y=double(y(:)); b=double(b(:)); h=double(h(:));
% Each contribution is paired hybrid-minus-baseline correctness.
z=double(h==y)-double(b==y); c0=find(y==0); c1=find(y==1);
delta=.5*(mean(z(c0))+mean(z(c1))); boots=zeros(cfg.Bootstrap,1);
for j=1:cfg.Bootstrap
    a=c0(randi(numel(c0),numel(c0),1)); c=c1(randi(numel(c1),numel(c1),1));
    boots(j)=.5*(mean(z(a))+mean(z(c)));
end
ci=prctile(boots,[2.5 97.5]); lo=ci(1); hi=ci(2); count=0;
for j=1:cfg.Permutations
    signflip=2*(rand(numel(y),1)>.5)-1;
    perm=.5*(mean(z(c0).*signflip(c0))+mean(z(c1).*signflip(c1)));
    count=count+(perm>=delta-1e-12);
end
pvalue=(count+1)/(cfg.Permutations+1);
end

function [delta,lo,hi,pvalue]=globalPairedTest(tables,cfg,seed)
% Mean disease DeltaBA, preserving dependence from reused healthy controls
% and any subjects appearing in multiple patient disease comparisons.
rng(seed,'twister'); allIDs=strings(0,1);
for j=1:numel(tables), allIDs=union(allIDs,tables{j}.SubjectID); end
n=numel(allIDs); m=numel(tables); z=NaN(n,m); lab=z;
for j=1:m
    t=tables{j}; [~,loc]=ismember(t.SubjectID,allIDs);
    z(loc,j)=double(t.HybridPrediction==t.Label)-double(t.BaselinePrediction==t.Label);
    lab(loc,j)=t.Label;
end
% Membership/label strata preserve every disease's two classes and shared IDs.
pattern=repmat("-",n,1);
for j=1:m
    col=repmat("X",n,1); col(lab(:,j)==0)="H"; col(lab(:,j)==1)="P";
    pattern=pattern+col;
end
[~,~,g]=unique(pattern); strata=cell(max(g),1);
for k=1:numel(strata), strata{k}=find(g==k); end
weights=ones(n,1); delta=globalDelta(z,lab,weights);
boots=zeros(cfg.Bootstrap,1);
for b=1:cfg.Bootstrap
    weights=zeros(n,1);
    for k=1:numel(strata)
        ix=strata{k}; draw=ix(randi(numel(ix),numel(ix),1));
        weights=weights+accumarray(draw,1,[n,1]);
    end
    boots(b)=globalDelta(z,lab,weights);
end
ci=prctctileSafe(boots); lo=ci(1); hi=ci(2); count=0;
for p=1:cfg.Permutations
    signs=2*(rand(n,1)>.5)-1;
    perm=globalDelta(z.*signs,lab,ones(n,1));
    count=count+(perm>=delta-1e-12);
end
pvalue=(count+1)/(cfg.Permutations+1);
end

function ci=prctctileSafe(values)
assert(all(isfinite(values)),'Undefined global bootstrap statistic.');
ci=prctile(values,[2.5,97.5]);
end

function value=globalDelta(z,lab,w)
v=zeros(1,size(z,2));
for j=1:size(z,2)
    a=lab(:,j)==0; b=lab(:,j)==1;
    assert(sum(w(a))>0 && sum(w(b))>0);
    v(j)=.5*(sum(w(a).*z(a,j))/sum(w(a))+sum(w(b).*z(b,j))/sum(w(b)));
end
value=mean(v);
end

function adjusted=holm(p,planned)
p=p(:); [sorted,ix]=sort(p); adjusted=zeros(size(p)); prior=0;
for j=1:numel(p)
    prior=max(prior,min(1,(planned-j+1)*sorted(j))); adjusted(ix(j))=prior;
end
end

function printResult(r,seconds)
for j=1:height(r.Summary)
    a=r.Summary(j,:);
    fprintf('%s | %s | %s | %s | k=%d | BA=%.2f +/- %.2f%% | Accuracy=%.2f%% | Sensitivity=%.2f%% | Specificity=%.2f%% | Time=%.1fs\n', ...
        a.Disease,a.Scenario,a.Classifier,a.Ranker,a.FeatureCount, ...
        100*a.CV_MeanBA,100*a.CV_StdBA,100*a.TestAccuracy, ...
        100*a.Sensitivity,100*a.Specificity,seconds);
end
end

function atomicSave(file,name,value)
folder=fileparts(file); if ~isfolder(folder), mkdir(folder); end
tmp=[tempname(folder) '.mat']; payload=struct(); payload.(name)=value;
save(tmp,'-struct','payload','-v7.3'); movefile(tmp,file,'f');
end

function atomicTable(t,file,sheet)
folder=fileparts(file); if ~isfolder(folder), mkdir(folder); end
tmp=[tempname(folder) '.xlsx']; writetable(t,tmp,'Sheet',sheet); movefile(tmp,file,'f');
end

function s=safeName(value)
s=char(regexprep(string(value),'[^A-Za-z0-9_-]','_'));
if ~strcmp(s,char(value))
    suffix=hashBytes(unicode2native(char(value),'UTF-8')); s=[s '_' suffix(1:8)];
end
end

function sig=runSignature(cfg,files)
% Content hashing prevents reuse after a same-size/timestamp Excel change.
% Includes THIS source file, so altered logic cannot reuse old checkpoints.
copy=cfg; copy=rmfield(copy,{'Resume','UseParallel','OutputDir'});
payload=jsonencode(copy);
for j=1:numel(files), payload=[payload '|' fileSHA(files(j))]; end %#ok<AGROW>
payload=[payload '|' fileSHA(string(mfilename('fullpath'))+".m")];
sig=string(hashBytes(unicode2native(payload,'UTF-8')));
end

function h=fileSHA(file)
md=java.security.MessageDigest.getInstance('SHA-256');
fid=fopen(file,'rb'); assert(fid~=-1,'Cannot open %s.',file);
cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
while true
    block=fread(fid,1024*1024,'*uint8'); if isempty(block), break; end
    md.update(typecast(block(:),'int8'));
end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end

function h=hashBytes(bytes)
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(typecast(uint8(bytes(:)),'int8'));
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end

function writeRunProtocol(cfg,names)
t=table((1:43)',reshape(names.All,[],1), ...
    [repmat("Temporal",10,1);repmat("Frequency",33,1)], ...
    'VariableNames',{'Index','ActualExcelColumn','FeatureFamily'});
atomicTable(t,fullfile(cfg.OutputDir,'Feature_Mapping.xlsx'),'Mapping');
protocol=cfg; atomicSave(fullfile(cfg.OutputDir,'Run_Protocol.mat'),'protocol',protocol);
if cfg.QuickTest
    base=5*3*16*cfg.MaxFolds; nested=2*3*16*cfg.MaxFolds^2;
else
    base=(3*43+10+33)*16*cfg.MaxFolds; nested=(10+33)*16*cfg.MaxFolds^2;
end
workload=table(base,nested,2*cfg.MaxFolds+5,base+nested+2*cfg.MaxFolds+5, ...
    numel(cfg.Diseases)*(base+nested+2*cfg.MaxFolds+5), ...
    'VariableNames',{'MainCVFitsPerDisease','NestedVotingCVFitsPerDisease', ...
    'AdditionalFitsPerDisease','UpperScheduledFitsPerDisease','UpperScheduledFitsAllDiseases'});
atomicTable(workload,fullfile(cfg.OutputDir,'Computational_Workload.xlsx'),'Workload');
end

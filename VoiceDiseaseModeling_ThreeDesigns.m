function VoiceDiseaseModeling_ThreeDesigns(overrides)
% Complete self-contained MATLAB R2023b implementation of three designs.
% Default: one-repeat pilot. Set PilotMode=false for the planned 100 repeats.
% Accuracy is the selection metric. All evaluation units are SUBJECTS.
% Existing Test results have been seen: this is exploratory internal validation.
cfg=configuration();
if nargin>0
    fields=fieldnames(overrides);
    for j=1:numel(fields),assert(isfield(cfg,fields{j}),'Unknown setting %s.',fields{j});cfg.(fields{j})=overrides.(fields{j});end
end
cfg.Diseases=reshape(string(cfg.Diseases),1,[]);
if cfg.Stage=="selftest",selftest(cfg);return;end
assert(ismember(cfg.Stage,["training","all"]),'Invalid Stage.');
assert(license('test','Statistics_Toolbox'),'Statistics and Machine Learning Toolbox required.');
assert(cfg.TestFraction==.20,'This protocol preserves subject-level 80/20.');
if cfg.QuickTest
    cfg.MaxFolds=2;cfg.NumTrees=8;cfg.NCAIterations=10;cfg.MaxTrainFramesPerSubject=9;
    cfg.OutputDir=fullfile(cfg.OutputDir,'QUICK_TEST_NOT_FOR_INFERENCE');
end
if isempty(cfg.RepeatIndices)
    if cfg.PilotMode,repeats=1;else,repeats=1:cfg.NumRepeats;end
else,repeats=cfg.RepeatIndices;end
assert(all(repeats>=1&repeats<=cfg.NumRepeats&repeats==fix(repeats)));
if ~isfolder(cfg.OutputDir),mkdir(cfg.OutputDir);end
if cfg.UseParallel
    assert(license('test','Distrib_Computing_Toolbox'),'Parallel toolbox required.');
    pool=gcp('nocreate');if isempty(pool),parpool('local',2);else,assert(pool.NumWorkers==2);end
end
routed=cfg;for field={'OutputDir','WarmStartRoot','UseParallel','Resume','Stage','RepeatIndices','PilotMode','ActiveDisease','DeadlineUTC'},routed=rmfield(routed,field);end
source=fileSHA(string(mfilename('fullpath'))+".m");
names=featureNames();allFiles=[featurePath(cfg,25,"Healthy"),featurePath(cfg,75,"Healthy")];
for disease=cfg.Diseases,allFiles=[allFiles,featurePath(cfg,25,disease),featurePath(cfg,75,disease)];end %#ok<AGROW>
manifest=string(jsonencode(routed))+source;
for file=allFiles,assert(isfile(file),'Missing input: %s',file);manifest=manifest+fileSHA(file);end
signature=string(hashBytes(unicode2native(char(manifest),'UTF-8')));
protocolFile=fullfile(cfg.OutputDir,'Study_Protocol.mat');
if isfile(protocolFile)
    saved=load(protocolFile,'protocol');assert(saved.protocol.Signature==signature,'Different code/data/settings: use a new OutputDir.');
else
    protocol=struct('Signature',signature,'Configuration',cfg,'SourceSHA',source,'Role',"Exploratory after prior Test inspection");
    atomicSave(protocolFile,'protocol',protocol);
end
cfg.DeadlineUTC=posixtime(datetime('now','TimeZone','UTC'))+cfg.RunBudgetHours*3600;
healthy=loadPair(cfg,"Healthy",0,names);status=table();paused=false;
for repeat=repeats
    rcfg=cfg;rcfg.Seed=cfg.Seed+10000*(repeat-1);rcfg.RepeatIndex=repeat;
    for disease=cfg.Diseases
        if strlength(cfg.ActiveDisease)>0&&disease~=cfg.ActiveDisease,continue;end
        location=fullfile(cfg.OutputDir,sprintf('Repeat_%03d',repeat),safeName(disease));if ~isfolder(location),mkdir(location);end
        complete=fullfile(location,'COMPLETE.mat');caseTimer=tic;
        try
            budget(cfg);
            if cfg.Resume&&isfile(complete),saved=load(complete,'result');result=saved.result;
            else
                patient=loadPair(cfg,disease,1,names);
                result=runThree(healthy,patient,disease,names,rcfg,location);
                if cfg.Stage=="all",result.ElapsedSeconds=toc(caseTimer);atomicSave(complete,'result',result);end
            end
            if cfg.Stage=="all",exportCase(result,location,cfg);printSelected(result);end %#ok<AGROW>
            status=[status;table(repeat,disease,"Completed_"+cfg.Stage,"",'VariableNames',{'Repeat','Disease','Status','Detail'})]; %#ok<AGROW>
        catch err
            failure=struct('Identifier',err.identifier,'Report',getReport(err,'extended','hyperlinks','off'));
            atomicSave(fullfile(location,'LastFailure.mat'),'failure',failure);
            state="Failed";if strcmp(err.identifier,'Voice:Budget'),state="PausedBudget";paused=true;end
            status=[status;table(repeat,disease,state,string(err.message),'VariableNames',{'Repeat','Disease','Status','Detail'})]; %#ok<AGROW>
            if strcmp(getenv('GITHUB_ACTIONS'),'true'),printError(failure.Report);end
        end
        atomicTable(status,fullfile(cfg.OutputDir,'Run_Status.xlsx'),'Status');
        if paused,break;end
    end
    if paused,break;end
end
if cfg.Stage=="all",aggregateResults(cfg);end
end

function cfg=configuration
cfg=struct('InputDir','D:\SVD\meta\svd_wav\Features_Final_CycleFallback_HNRfix', ...
    'OutputDir','D:\SVD\meta\svd_wav\ModelResults_ThreeDesigns', ...
    'Diseases',["Rekurrensparese","Hyperfunktionelle","Laryngitis",("Reinke "+string(char(214))+"dem"),"Spasmodische Dysphonie","Stimmlippenpolyp"], ...
    'Seed',1405,'TestFraction',.20,'MaxFolds',10,'NumRepeats',100,'RepeatIndices',[], ...
    'PilotMode',true,'QuickTest',false,'UseParallel',true,'Resume',true,'Stage',"all", ...
    'ActiveDisease',"",'MaxTrainFramesPerSubject',30,'RunBudgetHours',24,'DeadlineUTC',Inf, ...
    'Classifiers',["SVM_Linear","SVM_Quadratic","SVM_RBF","KNN3","KNN5","RandomForest","NaiveBayes"], ...
    'Rankers',["Ttest","CohenD","ReliefF","NCA"],'NumTrees',40,'MinLeafSize',5, ...
    'KNNNeighbors',5,'SVMBackend','certified_primal','KernelCacheMB',256,'NCAIterations',60, ...
    'ReliefNeighbors',10,'CalibrationFraction',.20,'TimeToleranceSec',.0051,'EqualityTolerance',1e-10, ...
    'IncludeEqualCount',true,'AdaptiveFeatureSelection',false,'WindowProbeFolds',10, ...
    'VotingAlphas',0:.1:1,'WarmStartRoot',"",'RepeatIndex',1,'Alpha',.05, ...
    'ProtocolVersion',"ThreeDesigns_Accuracy_Train30_v1");
end

function result=runThree(h,p,disease,names,cfg,cache)
a=[h.A(h.IA,:);p.A(p.IA,:)];b=[h.B(h.IB,:);p.B(p.IB,:)];
assert(isequal(a.SubjectID,b.SubjectID)&&isequal(a.FileName,b.FileName));
assert(isempty(intersect(unique(h.A.SubjectID),unique(p.A.SubjectID))));
[ids,~,s]=unique(a.SubjectID,'sorted');y=accumarray(s,double(a.Label),[],@mean);assert(all(y==0|y==1));
[train,test]=holdoutSubjects(y,cfg.TestFraction,cfg.Seed);fold=zeros(numel(ids),1);
fold(train)=groupFolds(y(train),cfg.MaxFolds,cfg.Seed+1);
cohort=table(ids,y,train,test,fold,'VariableNames',{'SubjectID','Label','Train','Test','Fold'});
atomicTable(cohort,fullfile(cache,'SubjectSplit.xlsx'),'SubjectSplit');atomicTable([h.Log;p.Log],fullfile(cache,'Matching.xlsx'),'Matching');
x25=double(a{:,cellstr(names.All)});x75=double(b{:,cellstr(names.All)});
base={makeData(x25,s,y,ids,names.All),makeData(x75,s,y,ids,names.All), ...
    makeData([x25(:,1:10),x75(:,11:43)],s,y,ids,["T25_"+names.Time,"F75_"+names.Frequency]), ...
    makeData(x25(:,1:10),s,y,ids,"T25_"+names.Time),makeData(x75(:,11:43),s,y,ids,"F75_"+names.Frequency)};
labels=["S1_25ms","S2_75ms","S3A_Hybrid","Time25_Branch","Frequency75_Branch"];
for j=1:5,base{j}.AdaptiveMode="fixed";base{j}.Token=disease;end
adaptive=makeData([x25,x75],s,y,ids,names.All);adaptive.Token=disease;adaptive.AdaptiveMode="all";
adaptive25=adaptive;adaptive25.AdaptiveMode="25";adaptive75=adaptive;adaptive75.AdaptiveMode="75";
summary=table();predictions=table(ids(test),y(test),'VariableNames',{'SubjectID','Label'});
features=table();windows=table();finalRanks=table();searches={};searchNames=strings(0,1);choices=struct();bundles={};evaluation={};votingDetails=struct();equalSummaries=table();equalPredictions=predictions;
for design=1:3
    tag=["D1_FS","D2_NoFS","D3_Adaptive"];tag=tag(design);where=fullfile(cache,tag);
    if ~isfolder(where),mkdir(where);end
    useFS=(design==1)||(design==3&&cfg.AdaptiveFeatureSelection);
    data=base;
    if design==3,data(3:5)={adaptive,adaptive25,adaptive75};end
    sc=cell(5,1);
    for j=1:5
        if design==3&&j<=2
            control=choose(cfg.AdaptiveFeatureSelection,1,2);sc{j}=choices.("Search"+control){j};
        else
            sc{j}=searchCV(data{j},find(train),fold(train),cfg,fullfile(where,labels(j)),cfg.Seed+100*j,useFS,cache);
        end
        searches{end+1}=sc{j};searchNames(end+1)=tag+"_"+labels(j); %#ok<AGROW>
    end
    choices.("Search"+design)=sc;
    vote=selectVote(data{4},data{5},find(train),fold(train),sc{4},sc{5},cfg, ...
        fullfile(where,'Voting'),useFS,Inf,cache);
    best=cellfun(@(x)x.Best,sc,'UniformOutput',false);
    equal=struct();
    if cfg.IncludeEqualCount&&useFS
        [equal.K,equal.Choices,equal.Curve]=chooseBudget(sc,cfg);
        equal.Vote=selectVote(data{4},data{5},find(train),fold(train),sc{4},sc{5},cfg, ...
            fullfile(where,'EqualVoting'),useFS,equal.K,cache);
    end
    choices.(tag)=struct('Best',{best},'Vote',vote,'Equal',equal);
    votingDetails.(tag)=vote;
    % Choices for ALL designs are frozen before any Test prediction is made.
    evaluation{design}=struct('Data',{data},'Searches',{sc},'Best',{best},'Vote',vote,'Equal',equal,'Tag',tag);
end
atomicSave(fullfile(cache,'ALL_TRAINING_CHOICES_FROZEN.mat'),'choices',choices);
if cfg.Stage=="training",result=choices;return;end
for design=1:3
    item=evaluation{design};models=cell(5,1);equalModels=cell(5,1);
    for j=1:5
        key=item.Tag+"_"+labels(j);file=fullfile(cache,"Model_"+key+".mat");
        if cfg.Resume&&isfile(file),saved=load(file,'bundle');bundle=saved.bundle;
        else,bundle=fitFinal(item.Data{j},find(train),item.Best{j},cfg,cfg.Seed+10000+j);atomicSave(file,'bundle',bundle);end
        models{j}=bundle;bundles{end+1}=bundle; %#ok<AGROW>
        if isfield(item.Equal,'K')
            file=fullfile(cache,"Equal_Model_"+key+".mat");
            if cfg.Resume&&isfile(file),saved=load(file,'bundle');equalModels{j}=saved.bundle;
            else,equalModels{j}=fitFinal(item.Data{j},find(train),item.Equal.Choices{j},cfg,cfg.Seed+10000+j);atomicSave(file,'bundle',equalModels{j});end
        end
    end
    evaluation{design}.Models=models;evaluation{design}.EqualModels=equalModels;
end
% Test starts here, after every design and equal-count configuration is fixed.
for design=1:3
    item=evaluation{design};probabilities=cell(5,1);eqp=cell(5,1);
    for j=1:5
        model=item.Models{j};
        ranks=model.RankingDetails;ranks.Feature=reshape(model.WindowFeatures,[],1);
        ranks.Design=repmat(item.Tag,height(ranks),1);ranks.Scenario=repmat(labels(j),height(ranks),1);
        finalRanks=[finalRanks;ranks]; %#ok<AGROW>
        probabilities{j}=predictSubjects(model,item.Data{j},find(test));
        key=item.Tag+"_"+labels(j);predictions.(key+"_Probability")=probabilities{j};
        predictions.(key+"_Prediction")=probabilities{j}>=.5;
        if j<=3,summary=[summary;metricRow(disease,cfg.RepeatIndex,item.Tag,labels(j),item.Best{j}, ...
                numel(model.Columns),y(test),probabilities{j}>=.5,sum(train))];end %#ok<AGROW>
        if ~isempty(model.Features)
            f=table(repmat(item.Tag,numel(model.Features),1),repmat(labels(j),numel(model.Features),1), ...
                reshape(model.Features,[],1),'VariableNames',{'Design','Scenario','Feature'});features=[features;f]; %#ok<AGROW>
        end
        if design==3
            map=table(repmat(labels(j),numel(model.WindowFeatures),1),reshape(model.WindowFeatures,[],1), ...
                'VariableNames',{'Scenario','WindowAndFeature'});windows=[windows;map]; %#ok<AGROW>
        end
        if isfield(item.Equal,'K')
            eqp{j}=predictSubjects(item.EqualModels{j},item.Data{j},find(test));
            equalPredictions.(key+"_Probability")=eqp{j};equalPredictions.(key+"_Prediction")=eqp{j}>=.5;
            if j<=3,equalSummaries=[equalSummaries;metricRow(disease,cfg.RepeatIndex,item.Tag,"Equal_"+labels(j), ...
                    item.Equal.Choices{j},numel(item.EqualModels{j}.Columns),y(test),eqp{j}>=.5,sum(train))];end %#ok<AGROW>
        end
    end
    finalVote=item.Vote.FinalChoice;
    [vp,vh]=combineVote(probabilities{4},probabilities{5},finalVote);
    predictions.(item.Tag+"_S3B_Voting_Probability")=vp;predictions.(item.Tag+"_S3B_Voting_Prediction")=vh;
    vb=item.Vote.Best;vb.Classifier=item.Best{4}.Classifier+" + "+item.Best{5}.Classifier;
    vb.Ranker=item.Best{4}.Ranker+" + "+item.Best{5}.Ranker;
    summary=[summary;metricRow(disease,cfg.RepeatIndex,item.Tag,"S3B_Voting",vb, ...
        numel(item.Models{4}.Columns)+numel(item.Models{5}.Columns),y(test),vh,sum(train))];
    summary.VotingMethod(end)=finalVote.Method;summary.TimeWeight(end)=finalVote.Alpha;
    summary.VoteTimeFeatures(end)=numel(item.Models{4}.Columns);summary.VoteFrequencyFeatures(end)=numel(item.Models{5}.Columns);
    summary.GenuineVoting(end)=finalVote.TwoProbabilitySources&&summary.VoteTimeFeatures(end)>0&&summary.VoteFrequencyFeatures(end)>0; %#ok<AGROW>
    if isfield(item.Equal,'K')
        [vp,vh]=combineVote(eqp{4},eqp{5},item.Equal.Vote.FinalChoice);
        equalPredictions.(item.Tag+"_S3B_Voting_Probability")=vp;equalPredictions.(item.Tag+"_S3B_Voting_Prediction")=vh;
        vb=item.Equal.Vote.Best;vb.Classifier=item.Equal.Choices{4}.Classifier+" + "+item.Equal.Choices{5}.Classifier;
        vb.Ranker=item.Equal.Choices{4}.Ranker+" + "+item.Equal.Choices{5}.Ranker;
        equalSummaries=[equalSummaries;metricRow(disease,cfg.RepeatIndex,item.Tag,"Equal_S3B_Voting",vb,item.Equal.K,y(test),vh,sum(train))]; %#ok<AGROW>
    end
end
result=struct('Disease',disease,'Repeat',cfg.RepeatIndex,'Summary',summary,'Predictions',predictions, ...
    'EqualCountSummary',equalSummaries,'EqualPredictions',equalPredictions,'Cohort',cohort, ...
    'Matching',[h.Log;p.Log],'SelectedFeatures',features,'AdaptiveWindows',windows, ...
    'FinalRankings',finalRanks,'Searches',{searches},'SearchNames',searchNames,'Voting',votingDetails, ...
    'ExploratoryAfterPriorTestSeen',true,'TrainingFrameCap',cfg.MaxTrainFramesPerSubject);
end

function value=choose(condition,a,b)
if condition,value=a;else,value=b;end
end

function budget(cfg)
if posixtime(datetime('now','TimeZone','UTC'))>cfg.DeadlineUTC,error('Voice:Budget','Time budget reached; checkpoints retained.');end
end

function printError(report)
escaped=strrep(strrep(strrep(report,'%','%25'),char(13),'%0D'),char(10),'%0A');fprintf(2,'::error title=Three-design study::%s\n',escaped);
end

function printSelected(result)
for j=1:height(result.Summary)
    r=result.Summary(j,:);
    fprintf('%s | %s | %s | %s | %s | K=%d | CV Accuracy %.2f +/- %.2f%% | Accuracy %.2f%% | Sensitivity %.2f%% | Specificity %.2f%% | %.1f seconds\n', ...
        r.Disease,r.Design,r.Scenario,r.Classifier,r.Ranker,r.FeatureCount,100*r.CVMeanAccuracy,100*r.CVStdAccuracy, ...
        100*r.TestAccuracy,100*r.TestSensitivity,100*r.TestSpecificity,result.ElapsedSeconds);
end
end

function search=searchCV(d,subjects,fold,cfg,cache,seed,useFS,caseRoot)
if ~isfolder(cache),mkdir(cache);end
candidates=candidateTable(cfg,numel(d.Names),useFS);nfold=max(fold);parts=cell(nfold,1);todo=true(nfold,1);
for k=1:nfold
    file=fullfile(cache,sprintf('Fold_%02d.mat',k));
    if cfg.Resume&&isfile(file)
        saved=load(file,'part');parts{k}=saved.part;
        todo(k)=~all(parts{k}.Done | ~parts{k}.Available);
    elseif useFS&&d.AdaptiveMode=="fixed"&&~cfg.QuickTest
        imported=warmFold(d,subjects,fold,k,candidates,cfg,cache,seed,caseRoot);
        if ~isempty(imported),parts{k}=imported;atomicSave(file,'part',imported);todo(k)=false;end
    end
end
if cfg.UseParallel&&any(todo)&&~isempty(gcp('nocreate'))
    tasks=find(todo);fresh=cell(numel(tasks),1);
    try
        parfor t=1:numel(tasks)
            k=tasks(t);part=evaluateFold(d,subjects,fold,k,candidates,cfg,seed+k,fullfile(cache,sprintf('Fold_%02d_partial.mat',k)));
            atomicSave(fullfile(cache,sprintf('Fold_%02d.mat',k)),'part',part);fresh{t}=part;
        end
        parts(tasks)=fresh;todo(tasks)=false;
    catch err
        evidence=struct('Report',getReport(err,'extended','hyperlinks','off'),'Action',"Finish missing folds serially without another pool");
        atomicSave(fullfile(cache,'ParallelRecovery.mat'),'evidence',evidence);
        for k=tasks(:)'
            file=fullfile(cache,sprintf('Fold_%02d.mat',k));
            if isfile(file),saved=load(file,'part');parts{k}=saved.part;todo(k)=~all(parts{k}.Done|~parts{k}.Available);end
        end
    end
end
for k=find(todo)'
    part=evaluateFold(d,subjects,fold,k,candidates,cfg,seed+k,fullfile(cache,sprintf('Fold_%02d_partial.mat',k)));
    atomicSave(fullfile(cache,sprintf('Fold_%02d.mat',k)),'part',part);parts{k}=part;
end
fields=metricFields();values=nan(nfold,height(candidates),numel(fields));available=false(nfold,height(candidates));failures=table();rankings=table();
for k=1:nfold
    values(k,:,:)=reshape(parts{k}.Values,[1,height(candidates),numel(fields)]);available(k,:)=parts{k}.Available;
    failures=[failures;parts{k}.Failures];rankings=[rankings;normalizeRanks(parts{k}.Rankings)]; %#ok<AGROW>
end
for j=1:numel(fields)
    candidates.("Mean"+fields(j))=reshape(mean(values(:,:,j),1,'omitnan'),[],1);
    candidates.("Std"+fields(j))=reshape(std(values(:,:,j),0,1,'omitnan'),[],1);
end
candidates.Sensitivity=candidates.MeanSensitivity;
candidates.Valid=all(available,1)' & all(isfinite(values(:,:,1)),1)';
if ~isempty(failures)
    atomicTable(failures,fullfile(cache,'Candidate_Failures.xlsx'),'Failures');
    error('Voice:Candidate','A required candidate failed: %s',strjoin(unique(failures.Error)," | "));
end
assert(any(candidates.Valid),'No candidate is available in every fold; adaptive branch may be empty.');
best=bestRow(candidates,cfg);search=struct('Best',best,'Candidates',candidates,'Failures',failures, ...
    'FoldRankings',rankings,'FoldValues',values,'Subjects',subjects,'Fold',fold,'Seed',seed,'Cache',string(cache));
atomicTable(candidates,fullfile(cache,'CV_Candidates.xlsx'),'Candidates');
end

function t=candidateTable(cfg,p,useFS)
if useFS
    counts=1:p;if cfg.QuickTest,counts=unique([1,2,ceil(p/2),p]);counts=counts(counts<=p);end
    rankers=cfg.Rankers;
else,counts=0;rankers="None";end % K=0 means ALL current columns, with no ranker.
[C,R,K]=ndgrid(1:numel(cfg.Classifiers),1:numel(rankers),counts);
t=table(reshape(cfg.Classifiers(C(:)),[],1),reshape(rankers(R(:)),[],1),K(:), ...
    'VariableNames',{'Classifier','Ranker','K'});
end

function part=evaluateFold(d,subjects,fold,k,candidates,cfg,seed,checkpoint)
budget(cfg);training=subjects(fold~=k);validation=subjects(fold==k);
assert(isempty(intersect(training,validation)));
prepared=prepareFold(d,training,validation,cfg);
C=height(candidates);part=struct('Values',nan(C,numel(metricFields())),'Done',false(C,1), ...
    'Available',true(C,1),'Failures',emptyFailures(),'Rankings',table(),'ImportedLegacy',false);
if cfg.Resume&&isfile(checkpoint)
    saved=load(checkpoint,'part');assert(isequal(size(saved.part.Values),size(part.Values)));
    part.Values=saved.part.Values;part.Done=saved.part.Done;part.Available=saved.part.Available;
end
rankers=unique(candidates.Ranker,'stable')';
for ranker=rankers
    ri=find(cfg.Rankers==ranker,1);if isempty(ri),ri=0;end
    [order,score,detail]=rankFeatures(prepared.Z,prepared.Subjects,d.Y,ranker,cfg,seed+ri);
    detail.Feature=reshape(prepared.Names,[],1);detail.Fold=repmat(k,height(detail),1);detail.Ranker=repmat(ranker,height(detail),1);
    part.Rankings=[part.Rankings;detail]; %#ok<AGROW>
    [svmOrder,~,svmDetail]=rankFeatures(prepared.SVMZ(prepared.FitMask,:),prepared.Subjects(prepared.FitMask),d.Y,ranker,cfg,seed+ri);
    svmDetail.Feature=reshape(prepared.SVMNames,[],1);svmDetail.Fold=repmat(k,height(svmDetail),1);svmDetail.Ranker=repmat(ranker+"_SVMFitOnly",height(svmDetail),1);
    part.Rankings=[part.Rankings;svmDetail]; %#ok<AGROW>
    for c=1:numel(cfg.Classifiers)
        isSVM=startsWith(cfg.Classifiers(c),"SVM");activeOrder=order;z=prepared.Z;zv=prepared.V;
        if isSVM,activeOrder=svmOrder;z=prepared.SVMZ;zv=prepared.SVMV;end
        cases=find(candidates.Classifier==cfg.Classifiers(c)&candidates.Ranker==ranker);
        for ci=cases(:)'
            if part.Done(ci),continue;end
            budget(cfg);count=candidates.K(ci);actual=count;if count==0,actual=numel(activeOrder);end
            if actual>numel(activeOrder),part.Available(ci)=false;continue;end
            columns=activeOrder(1:actual);
            try
                rng(seed+100*ri+1000*c+actual,'twister');
                model=fitModel(z(:,columns),prepared.Y,prepared.Weights,cfg.Classifiers(c),prepared.FitMask,prepared.CalMask,prepared.Subjects,cfg);
                probability=aggregateScores(modelProbability(model,zv(:,columns)),d.S(prepared.ValRows),validation);
                m=metrics(d.Y(validation),probability>=.5);part.Values(ci,:)=metricVector(m,actual);part.Done(ci)=true;
            catch err
                part.Failures(end+1,:)={k,cfg.Classifiers(c),ranker,count,string(err.message)};
            end
            atomicSearchCheckpoint(checkpoint,part);
        end
    end
end
end

function p=prepareFold(d,training,validation,cfg)
[fitIDs,calIDs]=calibrationSplit(training,d.Y,cfg);rows=trainingRows(d,training,cfg);vr=find(ismember(d.S,validation));
if d.AdaptiveMode=="fixed"
    [normal,names]=inputMapping(d,training,cfg);[svm,svmNames]=inputMapping(d,fitIDs,cfg);
else
    % A common fit-only map makes the two adaptive voting groups disjoint,
    % even when their winning classifiers differ. Calibration labels are excluded.
    [normal,names]=inputMapping(d,fitIDs,cfg);svm=normal;svmNames=names;
end
fit=ismember(d.S(rows),fitIDs);cal=ismember(d.S(rows),calIDs);
pre=fitPreprocessor(d.X(rows,normal));spre=fitPreprocessor(d.X(rows(fit),svm));
y=d.Y(d.S(rows));
p=struct('Z',applyPreprocessor(d.X(rows,normal),pre),'V',applyPreprocessor(d.X(vr,normal),pre), ...
    'SVMZ',applyPreprocessor(d.X(rows,svm),spre),'SVMV',applyPreprocessor(d.X(vr,svm),spre), ...
    'Names',names,'SVMNames',svmNames,'Subjects',d.S(rows),'Y',y,'Weights',subjectWeights(d.S(rows),y), ...
    'FitMask',fit,'CalMask',cal,'ValRows',vr,'Rows',rows,'Pre',pre,'SVMPre',spre,'Input',normal,'SVMInput',svm);
end

function [probability,model]=replayCandidate(d,training,validation,best,cfg,seed)
p=prepareFold(d,training,validation,cfg);ri=find(cfg.Rankers==best.Ranker,1);if isempty(ri),ri=0;end
ci=find(cfg.Classifiers==best.Classifier,1);z=p.Z;v=p.V;subjects=p.Subjects;
if startsWith(best.Classifier,"SVM"),z=p.SVMZ;v=p.SVMV;mask=p.FitMask;else,mask=true(numel(subjects),1);end
[order,~]=rankFeatures(z(mask,:),subjects(mask),d.Y,best.Ranker,cfg,seed+ri);
actual=best.K;if actual==0,actual=numel(order);end;assert(actual<=numel(order));cols=order(1:actual);
rng(seed+100*ri+1000*ci+actual,'twister');
model=fitModel(z(:,cols),p.Y,p.Weights,best.Classifier,p.FitMask,p.CalMask,subjects,cfg);
probability=aggregateScores(modelProbability(model,v(:,cols)),d.S(p.ValRows),validation);
end

function bundle=fitFinal(d,subjects,best,cfg,seed)
budget(cfg);p=prepareFold(d,subjects,zeros(0,1),cfg);z=p.Z;names=p.Names;input=p.Input;pre=p.Pre;mask=true(numel(p.Subjects),1);
if startsWith(best.Classifier,"SVM"),z=p.SVMZ;names=p.SVMNames;input=p.SVMInput;pre=p.SVMPre;mask=p.FitMask;end
[order,score,detail]=rankFeatures(z(mask,:),p.Subjects(mask),d.Y,best.Ranker,cfg,seed);
actual=best.K;if actual==0,actual=numel(order);end;assert(actual<=numel(order));columns=order(1:actual);
rng(seed,'twister');model=fitModel(z(:,columns),p.Y,p.Weights,best.Classifier,p.FitMask,p.CalMask,p.Subjects,cfg);
bundle=struct('Model',model,'Preprocessor',pre,'InputColumns',input,'Columns',columns, ...
    'Features',names(columns),'WindowFeatures',names,'RankingOrder',order,'RankingScores',score,'RankingDetails',detail, ...
    'UsedTrainingSubjects',d.IDs(subjects),'Choice',best, ...
    'FitSubjectIDs',d.IDs(unique(p.Subjects(p.FitMask))),'CalibrationSubjectIDs',d.IDs(unique(p.Subjects(p.CalMask))), ...
    'TrainingFrameCount',numel(p.Subjects));
end

function p=predictSubjects(bundle,d,subjects)
rows=find(ismember(d.S,subjects));z=applyPreprocessor(d.X(rows,bundle.InputColumns),bundle.Preprocessor);
p=aggregateScores(modelProbability(bundle.Model,z(:,bundle.Columns)),d.S(rows),subjects);
end

function p=chosenOOF(d,search,best,cfg)
key=safeName(best.Classifier+"_"+best.Ranker+"_"+best.K);file=fullfile(search.Cache,"OOF_"+key+".mat");
if cfg.Resume&&isfile(file),saved=load(file,'p');p=saved.p;return;end
p=nan(numel(search.Subjects),1);
for k=1:max(search.Fold)
    tr=search.Subjects(search.Fold~=k);va=search.Subjects(search.Fold==k);
    score=replayCandidate(d,tr,va,best,cfg,search.Seed+k);p(search.Fold==k)=score;
end
assert(all(isfinite(p)));atomicSave(file,'p',p);
end

function part=warmFold(d,subjects,fold,k,candidates,cfg,cache,seed,caseRoot)
part=[];if cfg.RepeatIndex~=1||strlength(cfg.WarmStartRoot)==0,return;end
marker=[filesep 'D1_FS' filesep];position=strfind(char(cache),marker);if isempty(position),return;end
suffix=char(cache);suffix=suffix(position(end)+numel(marker):end);
root=char(cfg.WarmStartRoot);assert(isfolder(root),'WarmStartRoot missing.');
spec=jsondecode(fileread(fullfile(root,'Experiment_Specification.json')));
assert(string(spec.Disease)==d.Token && spec.TrainingFrameCap==cfg.MaxTrainFramesPerSubject);
assert(string(spec.NumericalSourceSHA)=="31f4f7293f037a717f975aa8e70e2a2d4dd6eba9dfbc31d2ccc693915d0c6293",'Unrecognized previous numerical protocol.');
assert(string(spec.InputManifestSHA)==string(fileSHA(fullfile(cfg.InputDir,'manifest.json'))),'Warm inputs changed.');
old=load(fullfile(root,'Run_Protocol.mat'),'protocol');old=old.protocol;
for name=["Seed","MaxFolds","NumTrees","MinLeafSize","NCAIterations","ReliefNeighbors","CalibrationFraction","MaxTrainFramesPerSubject","SVMBackend","Classifiers","Rankers"]
    assert(isequal(string(old.(name)),string(cfg.(name))),'Warm numerical parameter mismatch: %s.',name);
end
split=dir(fullfile(root,'**','SubjectSplit.xlsx'));assert(numel(split)==1);
opts=detectImportOptions(fullfile(split.folder,split.name));opts=setvartype(opts,'SubjectID','string');previous=readtable(fullfile(split.folder,split.name),opts);
current=readtable(fullfile(caseRoot,'SubjectSplit.xlsx'),'TextType','string');current.SubjectID=string(current.SubjectID);
assert(isequal(string(previous.SubjectID),current.SubjectID)&&isequal(double(previous.Label),double(current.Label))&& ...
    isequal(logical(previous.Train),logical(current.Train))&&isequal(double(previous.Fold),double(current.Fold)),'Warm cohort mismatch.');
legacy=fileparts(fullfile(split.folder,split.name));file=fullfile(legacy,suffix,sprintf('Fold_%02d.mat',k));
if ~isfile(file),return;end
saved=load(file,'part');oldpart=saved.part;assert(numel(oldpart.BA)==height(candidates)&&all(isfinite(oldpart.BA)));
validation=subjects(fold==k);n0=sum(d.Y(validation)==0);n1=sum(d.Y(validation)==1);
values=zeros(height(candidates),numel(metricFields()));
for j=1:height(candidates)
    sensitivity=oldpart.Sensitivity(j);specificity=2*oldpart.BA(j)-sensitivity;
    tp=round(sensitivity*n1);tn=round(specificity*n0);
    assert(abs(tp-sensitivity*n1)<1e-7&&abs(tn-specificity*n0)<1e-7,'Nonintegral warm confusion counts.');
    m=metrics([ones(n1,1);zeros(n0,1)],[true(tp,1);false(n1-tp,1);false(tn,1);true(n0-tn,1)]);
    values(j,:)=metricVector(m,candidates.K(j));
end
% Replay one cheap candidate for each classifier in the first main fold.
% All learner/preprocessor/ranker numerics are retained from the old source.
if k==1&&endsWith(string(cache),"S1_25ms")
    training=subjects(fold~=k);
    for c=1:numel(cfg.Classifiers)
        j=find(candidates.Classifier==cfg.Classifiers(c)&candidates.Ranker=="Ttest"&candidates.K==1,1);
        predicted=replayCandidate(d,training,validation,table2struct(candidates(j,:)),cfg,seed+k);
        m=metrics(d.Y(validation),predicted>=.5);
        assert(abs(m.BalancedAccuracy-oldpart.BA(j))<1e-10&&abs(m.Sensitivity-oldpart.Sensitivity(j))<1e-10, ...
            'Warm replay mismatch for %s.',cfg.Classifiers(c));
    end
end
part=struct('Values',values,'Done',true(height(candidates),1),'Available',true(height(candidates),1), ...
    'Failures',emptyFailures(),'Rankings',oldpart.Rankings,'ImportedLegacy',true);
end

function t=emptyFailures
 t=table('Size',[0,5],'VariableTypes',{'double','string','string','double','string'}, ...
    'VariableNames',{'Fold','Classifier','Ranker','K','Error'});
end

function atomicSearchCheckpoint(file,complete)
part=struct('Values',complete.Values,'Done',complete.Done,'Available',complete.Available);
folder=fileparts(file);temporary=[tempname(folder) '.mat'];save(temporary,'part','-v7');movefile(temporary,file,'f');
end

function best=bestRow(t,cfg)
ix=find(t.Valid);assert(~isempty(ix));best=table2struct(t(ix(1),:));
for j=ix(:)'
    value=table2struct(t(j,:));if isBetter(value,best,cfg),best=value;end
end
end

function better=isBetter(a,b,cfg)
tol=cfg.EqualityTolerance;
if abs(a.MeanAccuracy-b.MeanAccuracy)>tol,better=a.MeanAccuracy>b.MeanAccuracy;return;end
if abs(a.StdAccuracy-b.StdAccuracy)>tol,better=a.StdAccuracy<b.StdAccuracy;return;end
if abs(a.Sensitivity-b.Sensitivity)>tol,better=a.Sensitivity>b.Sensitivity;return;end
better=a.K<b.K;
end

function names=metricFields
names=["Accuracy","BalancedAccuracy","Sensitivity","Specificity","F1","TP","TN","FP","FN","Subjects","FeatureCount"];
end

function vector=metricVector(m,count)
vector=[m.Accuracy,m.BalancedAccuracy,m.Sensitivity,m.Specificity,m.F1,m.TP,m.TN,m.FP,m.FN,m.TP+m.TN+m.FP+m.FN,count];
end

function t=normalizeRanks(t)
if ~ismember('PValue',t.Properties.VariableNames),t.PValue=nan(height(t),1);end
if ~ismember('CohenD',t.Properties.VariableNames),t.CohenD=nan(height(t),1);end
t=t(:,{'Fold','Ranker','Rank','Feature','Score','PValue','CohenD'});
end

function [columns,names]=inputMapping(d,ids,cfg)
if d.AdaptiveMode=="fixed",columns=1:size(d.X,2);names=d.Names;return;end
p=numel(d.Names);mapping=selectWindows(d,ids,cfg);
if d.AdaptiveMode=="all",original=1:p;
elseif d.AdaptiveMode=="25",original=find(mapping==1);
else,original=find(mapping==2);end
columns=original+(mapping(original)-1)*p;
names="W25_"+d.Names(original);names(mapping(original)==2)="W75_"+d.Names(original(mapping(original)==2));
end

function mapping=selectWindows(d,ids,cfg)
% Probe: independent per-feature Gaussian NB, trained on frames and scored
% by SUBJECT mean probability. This selects source windows, not feature count.
% Every call uses only this fitting partition; both adaptive branches share it.
persistent keys values
if isempty(keys),keys=strings(0,1);values={};end
identity=d.Token+"|"+strjoin(d.IDs(sort(ids)),",")+"|"+cfg.Seed+"|"+cfg.MaxTrainFramesPerSubject;
key=string(hashBytes(unicode2native(char(identity),'UTF-8')));
where=find(keys==key,1);if ~isempty(where),mapping=values{where};return;end
seed=cfg.Seed+mod(hex2dec(char(extractBetween(key,1,7))),10000000);
fold=groupFolds(d.Y(ids),min(cfg.MaxFolds,cfg.WindowProbeFolds),seed);
p=numel(d.Names);accuracy=nan(max(fold),2*p);sensitivity=accuracy;
for k=1:max(fold)
    training=ids(fold~=k);validation=ids(fold==k);rows=trainingRows(d,training,cfg);vr=find(ismember(d.S,validation));
    pre=fitPreprocessor(d.X(rows,:));x=applyPreprocessor(d.X(rows,:),pre);v=applyPreprocessor(d.X(vr,:),pre);
    y=d.Y(d.S(rows));weights=subjectWeights(d.S(rows),y);mu=zeros(2,2*p);variance=mu;
    for cls=0:1
        mask=y==cls;w=weights(mask)/sum(weights(mask));
        mu(cls+1,:)=sum(x(mask,:).*w,1);variance(cls+1,:)=max(sum((x(mask,:)-mu(cls+1,:)).^2.*w,1),1e-6);
    end
    log0=-.5*(log(variance(1,:))+(v-mu(1,:)).^2./variance(1,:));
    log1=-.5*(log(variance(2,:))+(v-mu(2,:)).^2./variance(2,:));probability=sigmoid(log1-log0);
    subjectProbability=zeros(numel(validation),2*p);
    for j=1:numel(validation),subjectProbability(j,:)=mean(probability(d.S(vr)==validation(j),:),1);end
    predicted=subjectProbability>=.5;label=d.Y(validation);
    accuracy(k,:)=mean(predicted==label,1);sensitivity(k,:)=mean(predicted(label==1,:),1);
end
ma=mean(accuracy,1);sa=std(accuracy,0,1);se=mean(sensitivity,1);mapping=ones(1,p);
for j=1:p
    a=struct('MeanAccuracy',ma(j),'StdAccuracy',sa(j),'Sensitivity',se(j),'K',1);
    b=struct('MeanAccuracy',ma(j+p),'StdAccuracy',sa(j+p),'Sensitivity',se(j+p),'K',1);
    if isBetter(b,a,cfg),mapping(j)=2;end
end
if numel(keys)>=300,keys=keys(151:end);values=values(151:end);end
keys(end+1)=key;values{end+1}=mapping;
end

function [order,score,detail]=rankFeatures(z,s,y,method,cfg,seed)
p=size(z,2);pv=nan(p,1);effect=pv;
if method=="None"
    order=1:p;score=nan(p,1);
else
    [order,score]=rankFeaturesLegacyScores(z,s,y,method,cfg,seed);
    [ids,~,g]=unique(s);u=zeros(numel(ids),p);
    for j=1:p,u(:,j)=accumarray(g,z(:,j),[],@mean);end
    label=y(ids);n0=sum(label==0);n1=sum(label==1);
    pooled=sqrt(((n0-1)*var(u(label==0,:),0,1)+(n1-1)*var(u(label==1,:),0,1))/(n0+n1-2));
    effect=((mean(u(label==1,:),1)-mean(u(label==0,:),1))./max(pooled,1e-12))';
    for j=1:p,[~,pv(j)]=ttest2(u(label==0,j),u(label==1,j),'Vartype','unequal');end
end
position=zeros(p,1);position(order)=(1:p)';
detail=table(position,score(:),pv,effect,'VariableNames',{'Rank','Score','PValue','CohenD'});
end

function best=bestAtCount(search,count,cfg)
t=search.Candidates;mask=t.K==count&t.Valid;assert(any(mask),'No valid candidate at k=%d.',count);best=bestRow(t(mask,:),cfg);
end

function [a,b]=branchBudget(time,freq,count,cfg)
found=false;
for kt=unique(time.Candidates.K)'
    kf=count-kt;if kf<1||~any(freq.Candidates.K==kf&freq.Candidates.Valid)||~any(time.Candidates.K==kt&time.Candidates.Valid),continue;end
    left=bestAtCount(time,kt,cfg);right=bestAtCount(freq,kf,cfg);
    value=struct('MeanAccuracy',(left.MeanAccuracy+right.MeanAccuracy)/2,'StdAccuracy',(left.StdAccuracy+right.StdAccuracy)/2, ...
        'Sensitivity',(left.Sensitivity+right.Sensitivity)/2,'K',count);
    if ~found||isBetter(value,winner,cfg),a=left;b=right;winner=value;found=true;end
end
assert(found,'No admissible split of the shared feature budget.');
end

function [count,choices,curve]=chooseBudget(searches,cfg)
common=intersect(intersect(searches{1}.Candidates.K,searches{2}.Candidates.K),searches{3}.Candidates.K);
common=unique(common(common>=2));found=false;curve=table();
for k=common(:)'
    try
        selected=cell(5,1);for j=1:3,selected{j}=bestAtCount(searches{j},k,cfg);end
        [selected{4},selected{5}]=branchBudget(searches{4},searches{5},k,cfg);
    catch,continue;end
    value=struct('MeanAccuracy',mean(cellfun(@(x)x.MeanAccuracy,selected(1:3))), ...
        'StdAccuracy',mean(cellfun(@(x)x.StdAccuracy,selected(1:3))), ...
        'Sensitivity',mean(cellfun(@(x)x.Sensitivity,selected(1:3))),'K',k);
    curve=[curve;struct2table(value)]; %#ok<AGROW>
    if ~found||isBetter(value,winner,cfg),winner=value;count=k;choices=selected;found=true;end
end
assert(found,'No common feature count for the equal-budget supplement.');
end

function vote=selectVote(time,freq,ids,fold,timeSearch,freqSearch,cfg,cache,useFS,featureBudget,caseRoot)
if ~isfolder(cache),mkdir(cache);end
root=fullfile(fileparts(cache),'Voting');kfold=max(fold);
allValues=nan(kfold,numel(metricFields()));outer=table();oof=nan(numel(ids),1);hardOOF=false(numel(ids),1);
for k=1:kfold
    file=fullfile(cache,sprintf('Outer_%02d.mat',k));
    if cfg.Resume&&isfile(file),saved=load(file,'part');part=saved.part;
    else
        tr=ids(fold~=k);va=ids(fold==k);inner=groupFolds(time.Y(tr),cfg.MaxFolds,cfg.Seed+20000+k);
        tc=searchCV(time,tr,inner,cfg,fullfile(root,sprintf('Outer_%02d_Time',k)),cfg.Seed+30000+k,useFS,caseRoot);
        fc=searchCV(freq,tr,inner,cfg,fullfile(root,sprintf('Outer_%02d_Frequency',k)),cfg.Seed+40000+k,useFS,caseRoot);
        a=tc.Best;b=fc.Best;if isfinite(featureBudget),[a,b]=branchBudget(tc,fc,featureBudget,cfg);end
        pt=chosenOOF(time,tc,a,cfg);pf=chosenOOF(freq,fc,b,cfg);
        choice=chooseVote(pt,pf,time.Y(tr),inner,a,b,cfg);
        tm=fitFinal(time,tr,a,cfg,cfg.Seed+50000+k);fm=fitFinal(freq,tr,b,cfg,cfg.Seed+60000+k);
        t=predictSubjects(tm,time,va);f=predictSubjects(fm,freq,va);[probability,prediction]=combineVote(t,f,choice);
        m=metrics(time.Y(va),prediction);nfeatures=numel(tm.Columns)+numel(fm.Columns);
        part=struct('Probability',probability,'Prediction',prediction,'Values',metricVector(m,nfeatures), ...
            'Choice',choice,'TimeChoice',a,'FrequencyChoice',b,'TimeCount',numel(tm.Columns),'FrequencyCount',numel(fm.Columns));
        atomicSave(file,'part',part);
    end
    allValues(k,:)=part.Values;oof(fold==k)=part.Probability;hardOOF(fold==k)=part.Prediction;
    outer=[outer;table(k,part.TimeChoice.Classifier,part.TimeChoice.Ranker,part.TimeCount, ...
        part.FrequencyChoice.Classifier,part.FrequencyChoice.Ranker,part.FrequencyCount, ...
        part.Choice.Method,part.Choice.Alpha,'VariableNames',{'Fold','TimeClassifier','TimeRanker','TimeFeatures', ...
        'FrequencyClassifier','FrequencyRanker','FrequencyFeatures','Method','TimeWeight'})]; %#ok<AGROW>
end
a=timeSearch.Best;b=freqSearch.Best;if isfinite(featureBudget),[a,b]=branchBudget(timeSearch,freqSearch,featureBudget,cfg);end
pt=chosenOOF(time,timeSearch,a,cfg);pf=chosenOOF(freq,freqSearch,b,cfg);
finalChoice=chooseVote(pt,pf,time.Y(ids),fold,a,b,cfg);
fields=metricFields();best=struct('Classifier',"IndependentBranches",'Ranker',"IndependentBranches",'K',0);
for j=1:numel(fields),best.("Mean"+fields(j))=mean(allValues(:,j));best.("Std"+fields(j))=std(allValues(:,j));end
best.Sensitivity=best.MeanSensitivity;
vote=struct('Best',best,'FinalChoice',finalChoice,'OuterChoices',outer, ...
    'TrainingOOF',table(time.IDs(ids),time.Y(ids),fold,oof,hardOOF, ...
    'VariableNames',{'SubjectID','Label','Fold','Probability','Prediction'}));
atomicSave(fullfile(cache,'Voting_Selected.mat'),'vote',vote);
end

function choice=chooseVote(pt,pf,y,fold,timeBest,freqBest,cfg)
prefer=~isBetter(freqBest,timeBest,cfg);
methods=["Soft","Weighted","Hard"];found=false;grid=table();
for method=methods
    alphas=.5;if method=="Weighted",alphas=cfg.VotingAlphas;elseif method=="Hard",alphas=double(prefer);end
    for alpha=alphas
        candidate=struct('Method',method,'Alpha',alpha,'PreferTime',prefer);
        [~,h]=combineVote(pt,pf,candidate);accuracy=zeros(max(fold),1);sensitivity=accuracy;
        for k=1:max(fold),m=metrics(y(fold==k),h(fold==k));accuracy(k)=m.Accuracy;sensitivity(k)=m.Sensitivity;end
        score=struct('MeanAccuracy',mean(accuracy),'StdAccuracy',std(accuracy),'Sensitivity',mean(sensitivity),'K',0);
        row=table(method,alpha,score.MeanAccuracy,score.StdAccuracy,score.Sensitivity, ...
            'VariableNames',{'Method','TimeWeight','CVAccuracy','CVStdAccuracy','CVSensitivity'});grid=[grid;row]; %#ok<AGROW>
        better=~found;if found,better=isBetter(score,winner,cfg);end
        if found&&method=="Weighted"&&choice.Method=="Weighted"&& ...
                abs(score.MeanAccuracy-winner.MeanAccuracy)<cfg.EqualityTolerance&& ...
                abs(score.StdAccuracy-winner.StdAccuracy)<cfg.EqualityTolerance&& ...
                abs(score.Sensitivity-winner.Sensitivity)<cfg.EqualityTolerance
            better=abs(alpha-.5)<abs(choice.Alpha-.5);
        end
        if better,choice=candidate;winner=score;found=true;end
    end
end
choice.Grid=grid;choice.SelectionUsesTrainingOnly=true;
choice.TwoProbabilitySources=(choice.Method~="Hard"&&choice.Alpha>0&&choice.Alpha<1);
end

function [probability,prediction]=combineVote(time,freq,choice)
if choice.Method=="Hard"
    prediction=hardVote(time,freq,choice.PreferTime);
    if choice.PreferTime,probability=time;else,probability=freq;end
else
    probability=choice.Alpha*time+(1-choice.Alpha)*freq;prediction=probability>=.5;
end
end

function t=metricRow(disease,repeat,design,scenario,best,count,y,h,ntrain)
m=metrics(y,h);t=table(disease,repeat,design,scenario,string(best.Classifier),string(best.Ranker),count, ...
    ntrain,numel(y),numel(y)/ntrain,'VariableNames',{'Disease','Repeat','Design','Scenario','Classifier','Ranker','FeatureCount','TrainSubjects','TestSubjects','TestTrainRatio'});
t.VotingMethod="";t.TimeWeight=NaN;t.VoteTimeFeatures=NaN;t.VoteFrequencyFeatures=NaN;t.GenuineVoting=false;
for field=["Accuracy","BalancedAccuracy","Sensitivity","Specificity","F1"]
    t.("CVMean"+field)=best.("Mean"+field);t.("CVStd"+field)=best.("Std"+field);
end
for field=["Accuracy","BalancedAccuracy","Sensitivity","Specificity","F1","TP","TN","FP","FN"],t.("Test"+field)=m.(field);end
end

function exportCase(r,out,cfg)
file=fullfile(out,[safeName(r.Disease) '_Results.xlsx']);temporary=[tempname(out) '.xlsx'];
writetable(r.Summary,temporary,'Sheet','Summary');writetable(r.Predictions,temporary,'Sheet','SubjectPredictions');
writetable(r.Cohort,temporary,'Sheet','SubjectSplit');writetable(r.Matching,temporary,'Sheet','Matching');
writetable(r.SelectedFeatures,temporary,'Sheet','SelectedFeatures');writetable(r.AdaptiveWindows,temporary,'Sheet','AdaptiveWindows');
if ~isempty(r.FinalRankings),writetable(r.FinalRankings,temporary,'Sheet','FinalTrainingRanks');end
if ~isempty(r.EqualCountSummary)
    writetable(r.EqualCountSummary,temporary,'Sheet','EqualCountSummary');writetable(r.EqualPredictions,temporary,'Sheet','EqualCountPredictions');
end
for j=1:numel(r.Searches)
    sc=r.Searches{j};writetable(sc.Candidates,temporary,'Sheet',sprintf('CV_%02d',j));
    writetable(sc.FoldRankings,temporary,'Sheet',sprintf('FoldRanks_%02d',j));
    drawCurves(sc,fullfile(out,'Figures'),r.SearchNames(j),cfg);
end
index=table((1:numel(r.SearchNames))',reshape(r.SearchNames,[],1),'VariableNames',{'CVSheetNumber','DesignAndBranch'});writetable(index,temporary,'Sheet','CV_Index');
fields=fieldnames(r.Voting);
for j=1:numel(fields)
    vote=r.Voting.(fields{j});writetable(vote.OuterChoices,temporary,'Sheet',sprintf('VotingOuter_%d',j));
    writetable(vote.FinalChoice.Grid,temporary,'Sheet',sprintf('VotingWeights_%d',j));writetable(vote.TrainingOOF,temporary,'Sheet',sprintf('VotingOOF_%d',j));
    choice=vote.FinalChoice;choice=rmfield(choice,'Grid');selected=r.Summary(r.Summary.Design==string(fields{j})&r.Summary.Scenario=="S3B_Voting",:);
    choice.TimeFeatureCount=selected.VoteTimeFeatures;choice.FrequencyFeatureCount=selected.VoteFrequencyFeatures;
    choice.TwoProbabilitySources=selected.GenuineVoting;
    writetable(struct2table(choice),temporary,'Sheet',sprintf('VotingChoice_%d',j));
end
movefile(temporary,file,'f');
end

function drawCurves(search,out,titleText,cfg)
t=search.Candidates;if all(t.Ranker=="None"),return;end
if ~isfolder(out),mkdir(out);end
for ranker=unique(t.Ranker,'stable')'
    file=fullfile(out,[safeName(titleText+"_"+ranker) '.png']);if cfg.Resume&&isfile(file),continue;end
    figureHandle=figure('Visible','off','Color','w','Position',[0,0,1200,700]);guard=onCleanup(@()close(figureHandle)); %#ok<NASGU>
    hold on;colors=lines(numel(cfg.Classifiers));
    for j=1:numel(cfg.Classifiers)
        rows=t(t.Ranker==ranker&t.Classifier==cfg.Classifiers(j)&t.Valid,:);[~,order]=sort(rows.K);rows=rows(order,:);
        if isempty(rows),continue;end
        errorbar(rows.K,100*rows.MeanAccuracy,100*rows.StdAccuracy,'Color',colors(j,:),'DisplayName',cfg.Classifiers(j),'LineWidth',.8);
    end
    counts=unique(t.K(t.Ranker==ranker&t.Valid));envelope=nan(size(counts));deviation=envelope;
    for j=1:numel(counts),b=bestRow(t(t.Ranker==ranker&t.K==counts(j)&t.Valid,:),cfg);envelope(j)=b.MeanAccuracy;deviation(j)=b.StdAccuracy;end
    errorbar(counts,100*envelope,100*deviation,'k-o','LineWidth',2,'DisplayName','Best classifier at each K');
    b=bestRow(t(t.Ranker==ranker&t.Valid,:),cfg);plot(b.K,100*b.MeanAccuracy,'rp','MarkerSize',16,'MarkerFaceColor','r','DisplayName','Selected K');
    xlabel('Number of cumulative ranked features');ylabel('Subject CV Accuracy (%)');title(titleText+" | "+ranker,'Interpreter','none');
    grid on;legend('Location','eastoutside','Interpreter','none');ylim([0,100]);exportgraphics(figureHandle,file,'Resolution',150);clear guard;
end
end

function aggregateResults(cfg)
files=dir(fullfile(cfg.OutputDir,'Repeat_*','*','COMPLETE.mat'));summary=table();equal=table();predictions=table();
for j=1:numel(files)
    saved=load(fullfile(files(j).folder,files(j).name),'result');r=saved.result;
    summary=[summary;r.Summary];equal=[equal;r.EqualCountSummary]; %#ok<AGROW>
    p=r.Predictions;p.Disease=repmat(r.Disease,height(p),1);p.Repeat=repmat(r.Repeat,height(p),1);predictions=[predictions;p]; %#ok<AGROW>
end
if isempty(summary),return;end
book=fullfile(cfg.OutputDir,'All_Diseases_Summary.xlsx');temporary=[tempname(cfg.OutputDir) '.xlsx'];
writetable(summary,temporary,'Sheet','Summary');writetable(predictions,temporary,'Sheet','SubjectPredictions');
if ~isempty(equal),writetable(equal,temporary,'Sheet','EqualCountSummary');end
if isfile(fullfile(cfg.OutputDir,'Run_Status.xlsx')),s=readtable(fullfile(cfg.OutputDir,'Run_Status.xlsx'),'TextType','string');writetable(s,temporary,'Sheet','LastInvocationStatus');end
scope=table(cfg.NumRepeats,numel(unique(summary.Repeat)),cfg.PilotMode,true, ...
    "Exploratory internal validation; previously inspected Test; corrected resampling inference is approximate", ...
    'VariableNames',{'PlannedRepeats','AvailableRepeatIndices','PilotMode','PreviouslyInspectedTest','Interpretation'});
writetable(scope,temporary,'Sheet','StudyScope');movefile(temporary,book,'f');
[perDisease,overall,differences]=repeatedComparisons(summary,cfg);
book=fullfile(cfg.OutputDir,'Statistical_Hypothesis_Validation.xlsx');temporary=[tempname(cfg.OutputDir) '.xlsx'];
writetable(perDisease,temporary,'Sheet','PerDisease');writetable(overall,temporary,'Sheet','Overall');
writetable(differences,temporary,'Sheet','PairedRepeatDifferences');writetable(scope,temporary,'Sheet','StudyScope');movefile(temporary,book,'f');
report=struct('Role',"ExploratoryInternalValidation",'PlannedRepeats',cfg.NumRepeats,'Summary',table2struct(summary), ...
    'Statistics',table2struct(perDisease),'Overall',table2struct(overall));
fid=fopen(fullfile(cfg.OutputDir,'Final_Report.json'),'w');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(report));
end

function [perDisease,overall,differences]=repeatedComparisons(summary,cfg)
perDisease=table();overall=table();differences=table();
for design=["D1_FS","D2_NoFS","D3_Adaptive"]
    for hybrid=["S3A_Hybrid","S3B_Voting"]
        for single=["S1_25ms","S2_75ms"]
            for disease=cfg.Diseases
                h=summary(summary.Design==design&summary.Scenario==hybrid&summary.Disease==disease,:);
                s=summary(summary.Design==design&summary.Scenario==single&summary.Disease==disease,:);
                [repeat,ih,is]=intersect(h.Repeat,s.Repeat);if isempty(repeat),continue;end
                delta=h.TestAccuracy(ih)-s.TestAccuracy(is);ratio=h.TestTrainRatio(ih);
                row=correctedComparison(delta,ratio,cfg.Alpha);row.Disease=disease;row.Design=design;row.Hybrid=hybrid;row.SingleWindow=single;
                perDisease=[perDisease;struct2table(row)]; %#ok<AGROW>
                d=table(repmat(disease,numel(repeat),1),repmat(design,numel(repeat),1),repmat(hybrid,numel(repeat),1), ...
                    repmat(single,numel(repeat),1),repeat,delta,h.TestBalancedAccuracy(ih)-s.TestBalancedAccuracy(is), ...
                    h.TestSensitivity(ih)-s.TestSensitivity(is),ratio, ...
                    'VariableNames',{'Disease','Design','Hybrid','SingleWindow','Repeat','DeltaAccuracy','DeltaBA','DeltaSensitivity','TestTrainRatio'});
                differences=[differences;d]; %#ok<AGROW>
            end
            if isempty(differences),continue;end
            d=differences(differences.Design==design&differences.Hybrid==hybrid&differences.SingleWindow==single,:);
            delta=[];ratio=[];
            for repeat=unique(d.Repeat)'
                a=d(d.Repeat==repeat,:);if numel(unique(a.Disease))~=numel(cfg.Diseases),continue;end
                delta(end+1,1)=mean(a.DeltaAccuracy);ratio(end+1,1)=mean(a.TestTrainRatio); %#ok<AGROW>
            end
            row=correctedComparison(delta,ratio,cfg.Alpha);row.Design=design;row.Hybrid=hybrid;row.SingleWindow=single;overall=[overall;struct2table(row)]; %#ok<AGROW>
        end
    end
end
if ~isempty(perDisease)
    perDisease.P_Holm=holm(perDisease.P_TwoSided,12*numel(cfg.Diseases));
    perDisease.ExploratoryEvidence=perDisease.DeltaAccuracy>0&perDisease.CI95Lower>0&perDisease.P_Holm<cfg.Alpha;
    perDisease.IndependentConfirmation=false(height(perDisease),1);
end
if ~isempty(overall)
    overall.P_Holm=holm(overall.P_TwoSided,12);overall.ExploratoryEvidence=overall.DeltaAccuracy>0&overall.CI95Lower>0&overall.P_Holm<cfg.Alpha;
    overall.IndependentConfirmation=false(height(overall),1);
end
end

function r=correctedComparison(delta,ratio,alpha)
r=struct('Repeats',numel(delta),'DeltaAccuracy',NaN,'StdDeltaAccuracy',NaN,'CorrectedSE',NaN, ...
    'CI95Lower',NaN,'CI95Upper',NaN,'P_TwoSided',NaN,'InferenceStatus',"Insufficient repeats");
if isempty(delta),return;end
r.DeltaAccuracy=mean(delta);r.StdDeltaAccuracy=std(delta);
if numel(delta)<2,return;end
r.CorrectedSE=std(delta)*sqrt(1/numel(delta)+mean(ratio));
if r.CorrectedSE<eps,r.InferenceStatus="Zero empirical variance: p-value not estimable";return;end
critical=tinv(1-alpha/2,numel(delta)-1);r.CI95Lower=r.DeltaAccuracy-critical*r.CorrectedSE;r.CI95Upper=r.DeltaAccuracy+critical*r.CorrectedSE;
r.P_TwoSided=2*tcdf(-abs(r.DeltaAccuracy/r.CorrectedSE),numel(delta)-1);r.InferenceStatus="Approximate corrected resampled paired t; exploratory";
end

function adjusted=holm(p,planned)
adjusted=nan(size(p));ids=find(isfinite(p));[value,order]=sort(p(ids));previous=0;
for j=1:numel(order),previous=max(previous,(planned-j+1)*value(j));adjusted(ids(order(j)))=min(1,previous);end
end

function selftest(cfg)
m=metrics([0;0;1;1],[0;1;1;1]);assert(m.Accuracy==.75&&m.BalancedAccuracy==.75&&m.TP==2&&m.TN==1&&m.FP==1&&m.FN==0);
pt=[.9;.2;.4;.8];pf=[.1;.8;.7;.6];
for prefer=[true,false]
    c=struct('Method',"Hard",'Alpha',double(prefer),'PreferTime',prefer);[p,h]=combineVote(pt,pf,c);
    if prefer,assert(isequal(p,pt)&&isequal(h,pt>=.5));else,assert(isequal(p,pf)&&isequal(h,pf>=.5));end
end
r=correctedComparison(.1,.25,.05);assert(isnan(r.P_TwoSided)&&r.Repeats==1);
d=[.1;.03;-.02;.07];r=correctedComparison(d,.25*ones(4,1),.05);assert(abs(r.CorrectedSE-std(d)*sqrt(.5))<1e-12);
assert(all(holm([.01;.04],2)>=[.02;.04]-1e-12));
x=randn(10,0);model=fitModel(x,[zeros(5,1);ones(5,1)],ones(10,1),"NaiveBayes",true(10,1),false(10,1),(1:10)',cfg);
assert(all(modelProbability(model,x)==.5));
fprintf('THREE_DESIGNS_NUMERIC_TESTS_PASSED\n');
end

function n=featureNames()
n.Time=["ZCR","ZCR_Var","RMS","PeakAmp","Entropy","EnergyEntropy", ...
    "JitterLocal_CycleOrProxy","JitterRAP_CycleOrProxy","ShimmerLocal_CycleOrProxy","ShimmerAPQ5_CycleOrProxy"];
n.Frequency=["HNR","SpecCentroid","SpecRolloff","SpecFlux","SpecEntropy", ...
    "MFCC"+string(1:13),"DeltaMFCC"+string(1:13),"FormantF1","FormantF2"];
n.All=[n.Time,n.Frequency];
assert(numel(n.All)==43 && numel(unique(n.All))==43);
end


function p=featurePath(cfg,ms,group)
p=string(fullfile(cfg.InputDir,sprintf('FullFeats_%dms_%s.xlsx',ms,group)));
if ~isfile(p)
    packed=string(fullfile(cfg.InputDir,sprintf('FullFeats_%dms_%s.mat',ms,group)));
    if isfile(packed),p=packed;end
end
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
required=["SubjectID","FileName","Vowel","Label","FrameID",names.All];
if endsWith(path,".mat")
    packed=load(path);
    assert(isequal(reshape(string(packed.FeatureNames),1,[]),names.All),'Packed feature names differ.');
    assert(size(packed.X,2)==43,'Packed feature dimension differs.');
    t=table(string(packed.SubjectID(:)),string(packed.FileName(:)),string(packed.Vowel(:)), ...
        double(packed.Label(:)),string(packed.Disease(:)),double(packed.FrameID(:)), ...
        double(packed.FrameStartTime(:)),double(packed.FrameCenterTime(:)), ...
        'VariableNames',{'SubjectID','FileName','Vowel','Label','Disease','FrameID','FrameStartTime','FrameCenterTime'});
    t=[t,array2table(double(packed.X),'VariableNames',cellstr(names.All))];
else
    opts=detectImportOptions(path,'VariableNamingRule','preserve');
    assert(all(ismember(required,string(opts.VariableNames))), ...
        'Workbook %s does not match the audited schema.',path);
    opts=setvartype(opts,{'SubjectID','FileName','Vowel'},'string');t=readtable(path,opts);
end
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


function model=fitModel(x,y,w,kind,fitMask,calMask,subject,cfg)
if isempty(x),model=struct('Kind',"EmptyBranch");return;end
options=statset('UseParallel',false);
model.Kind=kind;
switch kind
    case {"SVM","SVM_Linear","SVM_Quadratic","SVM_RBF"}
        % SVM raw margins are NOT probabilities. The margin model uses only
        % fit subjects; a Platt sigmoid uses separate calibration subjects.
        % Never call fitPosterior's frame-random cross-validation.
        if ismember(kind,["SVM_Quadratic","SVM_RBF"])
            kernel="gaussian";kernelArgs={};scaleRule='auto';
            if kind=="SVM_Quadratic"
                kernel="polynomial";kernelArgs={'PolynomialOrder',2};
                % Dimension-normalized quadratic kernel (1 + x*z'/p)^2.
                % Fixed numerical convention; no response labels set the scale.
                scaleRule=sqrt(size(x,2));
            end
            svm=fitcsvm(x(fitMask,:),y(fitMask),'KernelFunction',char(kernel), ...
                kernelArgs{:},'KernelScale',scaleRule,'BoxConstraint',1, ...
                'Standardize',false,'ClassNames',[0;1],'Prior','uniform', ...
                'Weights',w(fitMask),'Solver','SMO','CacheSize',cfg.KernelCacheMB, ...
                'IterationLimit',1000000,'KKTTolerance',1e-3);
            initialInfo=svm.ConvergenceInfo;kernelScale=svm.KernelParameters.Scale;
            if ~initialInfo.Converged
                % Same kernel, scale, C, priors and weights; alternate optimizer.
                % Do not accept a nonconverged model or change its hypothesis.
                svm=fitcsvm(x(fitMask,:),y(fitMask),'KernelFunction',char(kernel), ...
                    kernelArgs{:},'KernelScale',kernelScale,'BoxConstraint',1, ...
                    'Standardize',false,'ClassNames',[0;1],'Prior','uniform', ...
                    'Weights',w(fitMask),'Solver','ISDA','CacheSize',cfg.KernelCacheMB, ...
                    'IterationLimit',1000000,'KKTTolerance',1e-3);
            end
            model.SolverInfo=struct('InitialSMO',initialInfo,'Final',svm.ConvergenceInfo, ...
                'KernelScale',kernelScale);
            assert(svm.ConvergenceInfo.Converged,'%s scale=%.6g did not converge. SMO: %s Final: %s', ...
                kind,kernelScale,jsonencode(initialInfo),jsonencode(svm.ConvergenceInfo));
        elseif strcmp(cfg.SVMBackend,'certified_primal')
            wf=w(fitMask);yf=y(fitMask);
            for cls=0:1,wf(yf==cls)=.5*wf(yf==cls)/sum(wf(yf==cls));end
            [svm,fitInfo]=fitCertifiedLinearSVM(x(fitMask,:),yf,wf,1/sum(fitMask));
            model.SolverInfo=fitInfo;
        elseif strcmp(cfg.SVMBackend,'linear_dual')
            % fitclinear minimizes weighted mean hinge loss + lambda*||b||^2/2.
            % lambda=1/(n*C) matches fitcsvm linear soft-margin C=1.
            wf=w(fitMask);yf=y(fitMask);
            for cls=0:1,wf(yf==cls)=.5*wf(yf==cls)/sum(wf(yf==cls));end
            [svm,fitInfo]=fitclinear(x(fitMask,:),yf,'Learner','svm', ...
                'Regularization','ridge','Solver','dual','Lambda',1/sum(fitMask), ...
                'IterationLimit',1e9,'PassLimit',1000,'DeltaGradientTolerance',1e-3, ...
                'ClassNames',[0;1],'Weights',wf);
            useFallback=~all(fitInfo.TerminationCode>0) || ...
                (isfield(cfg,'ForceSVMFallback') && cfg.ForceSVMFallback);
            if isfield(fitInfo,'Alpha'),fitInfo=rmfield(fitInfo,'Alpha');end
            fitInfo.Backend='linear_dual';
            if useFallback
                % Same linear C=1 problem, solved by the verified reference
                % solver. Never accept the nonconverged fast model.
                svm=fitcsvm(x(fitMask,:),yf,'KernelFunction','linear', ...
                    'BoxConstraint',1,'Standardize',false,'ClassNames',[0;1], ...
                    'Prior','uniform','Weights',w(fitMask),'Solver','SMO');
                assert(svm.ConvergenceInfo.Converged,'Reference SVM did not converge.');
                fitInfo.Backend='SMO_converged_fallback';
                fitInfo.ReferenceConvergence=svm.ConvergenceInfo;
            end
            model.SolverInfo=fitInfo;
        else
            svm=fitcsvm(x(fitMask,:),y(fitMask),'KernelFunction','linear', ...
                'BoxConstraint',1,'Standardize',false,'ClassNames',[0;1], ...
                'Prior','uniform','Weights',w(fitMask),'Solver','SMO');
        end
        if isstruct(svm) && isfield(svm,'Type') && strcmp(svm.Type,'certified_linear_svm')
            raw=x(calMask,:)*svm.Beta+svm.Bias;
        else
            [~,scores]=predict(svm,x(calMask,:));raw=scores(:,svm.ClassNames==1);
        end
        wc=subjectWeights(subject(calMask),y(calMask));
        model.Core=svm; model.Calibration=fitPlatt(raw,y(calMask),wc);
    case {"KNN","KNN3","KNN5"}
        neighbors=cfg.KNNNeighbors;
        if kind=="KNN3",neighbors=3;elseif kind=="KNN5",neighbors=5;end
        model.Core=fitcknn(x,y,'NumNeighbors',min(neighbors,size(x,1)), ...
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
    case "EmptyBranch"
        p=.5*ones(size(x,1),1);
    case {"SVM","SVM_Linear","SVM_Quadratic","SVM_RBF"}
        if isstruct(model.Core) && isfield(model.Core,'Type') && strcmp(model.Core.Type,'certified_linear_svm')
            margin=x*model.Core.Beta+model.Core.Bias;
        else
            [~,sc]=predict(model.Core,x);margin=sc(:,model.Core.ClassNames==1);
        end
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


function h=hardVote(pt,pf,preferTime)
a=pt>=.5; b=pf>=.5; h=a&b; disagreement=a~=b;
if preferTime, h(disagreement)=a(disagreement); else, h(disagreement)=b(disagreement); end
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


function [svm,info,alpha]=fitCertifiedLinearSVM(X,y,w,lambda)
% Same weighted linear hinge-loss objective as C=1 fitcsvm when lambda=1/n.
% Accept only an independently feasible hinge-loss dual lower bound.
started=tic;X=double(X);y=double(y(:));w=double(w(:));w=w/sum(w);
assert(all(isfinite(X),'all') && all(ismember(y,[0 1])) && all(w>0));
assert(lambda>0 && size(X,1)==numel(y));
[n,p]=size(X);signed=2*y-1;Z=[X,ones(n,1)];theta=zeros(p+1,1);
reg=[lambda*ones(p,1);0];total=0;tol=1e-5;converged=false;
for delta=[.1,.01,.001,.0001,.00001,.000001,.0000001]
    for iteration=1:120
        total=total+1;margin=signed.*(Z*theta);r=1-margin;
        slope=min(1,max(0,r/delta));a=w.*slope;
        g=reg.*theta-Z'*(signed.*a);
        if norm(g,Inf)<1e-9,break;end
        active=r>0 & r<delta;V=Z(active,:);
        H=V'*(V.*(w(active)/delta))+diag(reg)+1e-9*eye(p+1);
        scale=1./sqrt(max(diag(H),1e-12));Hs=(H.*scale).*scale';
        step=scale.*(Hs\(scale.*g));
        if norm(step)>100,step=step*(100/norm(step));end
        descent=g'*step;
        if ~isfinite(descent) || descent<=0,step=g;descent=g'*g;end
        old=svmSmoothObjective(Z,signed,w,lambda,theta,delta);fraction=1;accepted=false;
        for search=1:55
            proposal=theta-fraction*step;
            if svmSmoothObjective(Z,signed,w,lambda,proposal,delta)<=old-1e-4*fraction*descent
                accepted=true;break
            end
            fraction=fraction/2;
        end
        if ~accepted,break;end
        theta=proposal;
    end
    [primal,dual,alpha,feasibility]=svmHingeCertificate(X,signed,w,lambda,theta,delta);
    gap=primal-dual;
    assert(gap>=-1e-8 && feasibility<1e-10,'Invalid SVM dual certificate.');
    if gap<=tol,converged=true;break;end
end
svm=struct('Type','certified_linear_svm','Beta',theta(1:p),'Bias',theta(end),'ClassNames',[0;1]);
info=struct('Backend','certified_primal_hinge','Converged',converged, ...
    'Lambda',lambda,'PrimalObjective',primal,'DualObjective',dual, ...
    'PrimalDualGap',max(0,gap),'GapTolerance',tol,'DualFeasibility',feasibility, ...
    'NewtonIterations',total,'FinalSmoothing',delta,'Seconds',toc(started));
assert(converged,'Certified SVM did not reach its hinge primal-dual gap tolerance (gap %.6g).',gap);
end


function value=svmSmoothObjective(Z,y,w,lambda,theta,delta)
r=1-y.*(Z*theta);loss=zeros(size(r));high=r>=delta;middle=r>0 & ~high;
loss(high)=r(high)-delta/2;loss(middle)=r(middle).^2/(2*delta);
value=.5*lambda*sum(theta(1:end-1).^2)+sum(w.*loss);
end


function [primal,dual,a,feasibility]=svmHingeCertificate(X,y,w,lambda,theta,delta)
r=1-y.*(X*theta(1:end-1)+theta(end));a=w.*min(1,max(0,r/delta));
% Projection onto 0<=a<=w and y'*a=0 makes the dual point feasible.
lo=-max(w);hi=max(w);
for k=1:70
    t=(lo+hi)/2;trial=min(w,max(0,a+t*y));
    if sum(y.*trial)>0,hi=t;else,lo=t;end
end
a=min(w,max(0,a+((lo+hi)/2)*y));feasibility=abs(sum(y.*a));q=X'*(a.*y);
primal=.5*lambda*sum(theta(1:end-1).^2)+sum(w.*max(0,r));
dual=sum(a)-sum(q.^2)/(2*lambda);
end


function [order,score]=rankFeaturesLegacyScores(z,s,y,method,cfg,seed)
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


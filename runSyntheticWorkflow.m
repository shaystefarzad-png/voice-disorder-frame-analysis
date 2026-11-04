function runSyntheticWorkflow
% Integration test only. No patient data and no scientific inference.
root=fileparts(mfilename('fullpath'));
artifacts=fullfile(root,'synthetic-test-artifacts');
if ~isfolder(artifacts), mkdir(artifacts); end
work=tempname; mkdir(work); input=fullfile(work,'Features'); mkdir(input);
output=fullfile(artifacts,'ModelResults');
source=fileread(fullfile(root,'VoiceDiseaseModeling_Final.m'));
assert(contains(source,'function statisticalReport') && contains(source,'function writeRunProtocol'), ...
    'Uploaded modeling source is incomplete.');
% Only a disposable copy has its configuration changed; no modeling logic is
% replaced, and the repository source remains unchanged.
source=strrep(source,"cfg.InputDir = 'D:\SVD\meta\svd_wav\Features_Corrected';", ...
    "cfg.InputDir = '"+escapePath(input)+"';");
source=strrep(source,"cfg.OutputDir = 'D:\SVD\meta\svd_wav\ModelResults_Final';", ...
    "cfg.OutputDir = '"+escapePath(output)+"';");
source=regexprep(source,'cfg\.Diseases = \["Rekurrensparese"[\s\S]*?"Stimmlippenpolyp"\];', ...
    'cfg.Diseases = "Hyperfunktionelle";');
source=strrep(source,'cfg.QuickTest=false','cfg.QuickTest=true');
assert(contains(source,"cfg.InputDir = '"+escapePath(input)+"';") && ...
    contains(source,'cfg.Diseases = "Hyperfunktionelle";') && contains(source,'cfg.QuickTest=true'));
file=fullfile(work,'VoiceDiseaseModeling_Final.m');
fid=fopen(file,'w'); assert(fid~=-1); fwrite(fid,char(source),'char'); fclose(fid);
rng(1405,'twister'); makeSyntheticExcel(input);
old=pwd; cleanup=onCleanup(@()cd(old)); %#ok<NASGU>
cd(work); addpath(work,'-begin'); clear VoiceDiseaseModeling_Final;
resolved=which('VoiceDiseaseModeling_Final'); assert(strcmp(resolved,file));
log=evalc('VoiceDiseaseModeling_Final');
writeText(fullfile(artifacts,'MATLAB_execution_log.txt'),log);
qdir=fullfile(output,'QUICK_TEST_NOT_FOR_INFERENCE');
book=fullfile(qdir,'All_Diseases_Summary.xlsx');
assert(isfile(book),'The main pipeline did not produce the summary workbook.');
status=readtable(book,'Sheet','RunStatus','TextType','string');
if ~all(status.Status=="Completed")
    disp(status); fprintf('%s\n',log);
    reports=dir(fullfile(qdir,'**','LastFailure.mat'));
    for j=1:numel(reports)
        evidence=load(fullfile(reports(j).folder,reports(j).name),'failure');
        report=string(evidence.failure.Report);
        fprintf(2,'%s\n',report);
        escaped=replace(replace(replace(report,"%","%25"),char(13),"%0D"),char(10),"%0A");
        fprintf(2,'::error title=Modeling pipeline root cause::%s\n',escaped);
        summaryFile=getenv('GITHUB_STEP_SUMMARY');
        if ~isempty(summaryFile)
            writeText(summaryFile,"## Modeling failure details"+newline+"```text"+newline+report+newline+"```"+newline);
        end
    end
end
assert(all(status.Status=="Completed"), ...
    'Pipeline reported failure. Inspect RunStatus and LastFailure.mat artifacts.');
summary=readtable(book,'Sheet','Summary','TextType','string');
assert(height(summary)==4 && numel(unique(summary.Scenario))==4);
assert(all(isfinite(summary.TestBalancedAccuracy)) && all(summary.TestBalancedAccuracy>=0 & summary.TestBalancedAccuracy<=1));
diseaseBook=fullfile(qdir,'Hyperfunktionelle','Hyperfunktionelle_Results.xlsx');
cohort=readtable(diseaseBook,'Sheet','SubjectSplit','TextType','string');
assert(height(cohort)==60 && numel(unique(cohort.SubjectID))==60);
assert(~any(cohort.Train & cohort.Test) && all(cohort.Train | cohort.Test));
assert(all(cohort.Fold(cohort.Test)==0) && all(cohort.Fold(cohort.Train)>0));
assert(sum(cohort.Test)==12 && sum(cohort.Train)==48);
matching=readtable(diseaseBook,'Sheet','Matching','TextType','string');
valid=matching.CommonFrames>0;
assert(sum(valid)==180 && sum(matching.CommonFrames)==1260);
assert(all(matching.CommonFrames(valid)==min(matching.Frames25(valid),matching.Frames75(valid))));
assert(all(matching.MaxTimeErrorSec(valid)<=.0051));
assert(any(matching.Status=="Missing75"),'Unmatched recording handling was not exercised.');
assert(any(matching.Status=="TimeMismatchExcluded"),'Invalid timestamp matching was not exercised.');
pred=readtable(diseaseBook,'Sheet','SubjectPredictions','TextType','string');
assert(height(pred)==12 && numel(unique(pred.SubjectID))==12);
assert(isequal(sort(pred.SubjectID),sort(cohort.SubjectID(cohort.Test))));
for scenario=["S1_25ms","S2_75ms","S3A_Hybrid","S3B_Voting"]
    h=pred.(scenario+"_Prediction"); y=pred.Label;
    ba=.5*(mean(h(y==1)==1)+mean(h(y==0)==0));
    assert(abs(summary.TestBalancedAccuracy(summary.Scenario==scenario)-ba)<1e-12, ...
        'Summary BA is inconsistent with SUBJECT-level predictions.');
end
stats=readtable(fullfile(qdir,'Statistical_Hypothesis_Validation.xlsx'),'Sheet','PerDisease');
assert(height(stats)==2 && ~any(stats.HypothesisSupported));
assert(all(stats.P_Holm>=stats.P_OneSided-1e-12));
assert(all(stats.CI95_Lower<=stats.CI95_Upper));
% A resumed invocation must not rewrite completed fold/model checkpoints.
cacheFiles=dir(fullfile(qdir,'**','*.mat'));
keep=contains(string({cacheFiles.folder}),'cache_'); cacheFiles=cacheFiles(keep);
assert(~isempty(cacheFiles));
before=checkpointState(cacheFiles);
log=evalc('VoiceDiseaseModeling_Final');
writeText(fullfile(artifacts,'MATLAB_resume_log.txt'),log);
after=checkpointState(cacheFiles);
assert(isequal(before,after),'Resume retrained or changed an existing completed checkpoint.');
writeText(fullfile(artifacts,'PASS.txt'), ...
    'Synthetic integration checks passed. This does not validate the research hypothesis or real-data performance.');
fprintf('SYNTHETIC_INTEGRATION_TEST_PASSED\n');
end

function makeSyntheticExcel(input)
names=["ZCR","ZCR_Var","RMS","PeakAmp","Entropy","EnergyEntropy", ...
    "JitterProxy_local","JitterProxy_rap","ShimmerProxy_local","ShimmerProxy_apq", ...
    "HNR","SpecCentroid","SpecRolloff","SpecFlux","SpecEntropy", ...
    "MFCC"+string(1:13),"DeltaMFCC"+string(1:13),"FormantF1","FormantF2"];
for label=0:1
    if label==0, disease="Healthy"; else, disease="Hyperfunktionelle"; end
    short=table(); long=table();
    for person=1:30
        id=string(1000+person+1000*label); subject=randn(1,43);
        for vowel=["a","i","u"]
            filename=disease+"_"+id+"-"+vowel+"_n";
            for ms=[25,75]
                n=12; offset=.01246875;
                if ms==75, n=7; offset=.03746875; end
                start=(0:n-1)'*.01;
                % Neither window nor the hybrid is designed to win.
                x=.5*subject+.8*randn(n,43)+.4*label;
                x(rand(size(x))<.015)=NaN;
                t=table(repmat(filename,n,1),repmat(id,n,1),repmat(vowel,n,1), ...
                    repmat(label,n,1),repmat("synthetic",n,1),repmat(disease,n,1), ...
                    (1:n)',start,start+offset,true(n,1),true(n,1),ones(n,1), ...
                    'VariableNames',{'FileName','SubjectID','Vowel','Label','Gender','Disease', ...
                    'FrameID','FrameStartTime','FrameCenterTime','IsActive','IsVoiced','VoicingConfidence'});
                t=[t,array2table(x,'VariableNames',cellstr(names))];
                if ms==25, short=[short;t]; else, long=[long;t]; end %#ok<AGROW>
            end
        end
    end
    % Additional unmatched and deliberately shifted recordings must be logged
    % and excluded without preventing the valid cohort from completing.
    extra=short(1:12,:); extra.FileName(:)=disease+"_unmatched";
    short=[short;extra];
    bad25=short(1:12,:); bad25.FileName(:)=disease+"_bad_timing";
    bad75=long(1:7,:); bad75.FileName(:)=disease+"_bad_timing";
    bad75.FrameStartTime=bad75.FrameStartTime+100;
    bad75.FrameCenterTime=bad75.FrameCenterTime+100;
    short=[short;bad25]; long=[long;bad75];
    writetable(short,fullfile(input,"FullFeats_25ms_"+disease+".xlsx"));
    writetable(long,fullfile(input,"FullFeats_75ms_"+disease+".xlsx"));
end
end

function x=checkpointState(files)
x=zeros(numel(files),2);
for j=1:numel(files)
    file=fullfile(files(j).folder,files(j).name); now=dir(file);
    x(j,:)=[now.bytes,now.datenum];
end
end

function s=escapePath(path)
s=replace(string(path),"'","''");
end

function writeText(file,text)
fid=fopen(file,'w'); assert(fid~=-1); cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,char(text),'char');
end

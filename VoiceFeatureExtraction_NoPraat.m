function VoiceFeatureExtraction_NoPraat(mode,overrides)
% Standalone MATLAB R2023b extractor for corrected native-PCM WAV inputs.
% Run: VoiceFeatureExtraction_NoPraat
% All local functions are included below. Requires Signal Processing Toolbox.
% Cycle metrics: MATLAB YIN-guided acoustic peak tracking; no external executable.
% This independent estimator is NOT numerically equivalent to Praat.
if nargin<1,mode="run";end;if nargin<2,overrides=struct();end
cfg=vfConfiguration();fields=fieldnames(overrides);
for j=1:numel(fields),assert(isfield(cfg,fields{j}),'Unknown setting %s.',fields{j});cfg.(fields{j})=overrides.(fields{j});end
cfg.Diseases=reshape(string(cfg.Diseases),1,[]);
assert(license('test','Signal_Toolbox'),'Signal Processing Toolbox is required.');
if string(mode)=="selftest",vfSelfTest(cfg);return;end
assert(string(mode)=="run",'Mode must be run or selftest.');
assert(isfile(cfg.DatasetPath),'New metadata file not found: %s',cfg.DatasetPath);
T=readtable(cfg.DatasetPath,'VariableNamingRule','preserve','TextType','string');
[T,inputAudit]=vfValidateMetadata(T,cfg);
if ~isfolder(cfg.OutputFolder),mkdir(cfg.OutputFolder);end
cfg.EngineVersion="MATLAB_YIN_AcousticCycles_v1";cfg.EngineSHA256=vfHashText(cfg.EngineVersion);
assert(cfg.PerturbationMode=="cycle_with_proxy_fallback" && isequal(cfg.ProxyFallbackWindowsMs,25),'This version supports cycle metrics with 25ms-only insufficient-cycle fallback.');
assert(isequal(cfg.WindowMs,[25,75]) && cfg.TargetFs==16000,'This validated version uses 25/75ms at 16kHz.');
sourceHash=vfHashFile(string(mfilename('fullpath'))+".m");
if cfg.AutoSelfTest
    proofFile=fullfile(cfg.OutputFolder,'Startup_SelfTest.mat');proofSignature=vfHashText(sourceHash+cfg.EngineSHA256);
    passed=false;
    if isfile(proofFile),saved=load(proofFile,'proofSignature');passed=saved.proofSignature==proofSignature;end
    if ~passed
        fprintf('Running one-time SYNTHETIC startup checks before real extraction...\n');
        vfSelfTest(cfg);vfAtomicSave(proofFile,'proofSignature',proofSignature);
        fprintf('Startup checks PASSED. Beginning real recordings.\n');
    end
end
settings=cfg;settings.CodeSHA256=sourceHash;settings.FeatureNames=vfFeatureNames(cfg);settings.InputMetadataSHA256=vfHashFile(cfg.DatasetPath);
settings.ProxyEffectiveSupportMs=cfg.WindowMs+2*cfg.PerturbRadius*cfg.HopMs;
settings.CycleMetricPercentUnits=true;settings.CycleContext="Acoustic peaks tracked from native-rate whole recording; queries restricted to each frame interval";
settings.MinimumPointProcessIntervals=[2,3,3,6];
settings.CycleEstimator="YIN-guided positive acoustic peaks; quadratic sample refinement; not glottal closure or Praat equivalent";
settings.HNRMethod="Overlap-normalized raw-sample correlation with fractional sinc lag refinement";
settings.FallbackRule="Only when 25ms has too few complete pulse intervals and periodic support; never on other invalid-cycle failures";
settings.DiagnosticColumnsAreNotModelFeatures=true;
settings.NoSubjectSplitOrModelSelection=true;
protocolFile=fullfile(cfg.OutputFolder,'Run_Protocol.mat');
protocolSettings=settings;protocolSettings.Resume=false;protocolSignature=vfHashText(jsonencode(protocolSettings));
if isfile(protocolFile),prior=load(protocolFile,'protocolSignature');assert(prior.protocolSignature==protocolSignature,'Different input/settings/code in this folder. Choose a NEW OutputFolder.');end
vfAtomicSave(protocolFile,'protocolSignature',protocolSignature);
writetable(inputAudit,fullfile(cfg.OutputFolder,'Input_Metadata_QC.xlsx'));
map=table((1:43)',settings.FeatureNames(:),'VariableNames',{'FeatureIndex','FeatureName'});vfAtomicTable(map,fullfile(cfg.OutputFolder,'Model_Feature_List.xlsx'));
if cfg.SaveCycleOnly
    if ~isfolder(fullfile(cfg.OutputFolder,'CycleOnly')),mkdir(fullfile(cfg.OutputFolder,'CycleOnly'));end
    pureNames=settings.FeatureNames;pureNames(7:10)=["JitterLocal","JitterRAP","ShimmerLocal","ShimmerAPQ5"];
    pureMap=table((1:43)',pureNames(:),'VariableNames',{'FeatureIndex','FeatureName'});
    vfAtomicTable(pureMap,fullfile(cfg.OutputFolder,'CycleOnly','Model_Feature_List.xlsx'));
end
save(fullfile(cfg.OutputFolder,'Extraction_Settings.mat'),'settings');
vfWriteJSON(fullfile(cfg.OutputFolder,'Extraction_Settings.json'),settings);
allQC=table();allFiles=table();diseases=cfg.Diseases(ismember(cfg.Diseases,unique(T.Disease)));
for disease=diseases
    Td=T(T.Disease==disease,:);signature=vfGroupSignature(Td,settings);
    checkpoint=fullfile(cfg.OutputFolder,"Completed_"+disease+".mat");
    if cfg.Resume && isfile(checkpoint)
        previous=load(checkpoint,'state');assert(previous.state.Signature==signature,'Input/settings changed. Use a new OutputFolder.');
        if previous.state.Complete && all(isfile(previous.state.Outputs))
            allQC=[allQC;previous.state.QC];allFiles=[allFiles;previous.state.Files]; %#ok<AGROW>
            vfSaveQC(allQC,allFiles,cfg);fprintf('%s: reused completed outputs.\n',disease);continue;
        end
    end
    groupQC=table();groupFiles=table();outputs=strings(0,1);complete=true;
    for window=cfg.WindowMs
        n=round(window*cfg.TargetFs/1000);hop=round(cfg.HopMs*cfg.TargetFs/1000);
        bank=createMelFilterBank(cfg.TargetFs,cfg.NFFT,26,0,cfg.TargetFs/2);
        meta=cell(height(Td),1);feats=cell(height(Td),1);pureFeats=cell(height(Td),1);q=initializeQC(window,disease);q.ExpectedFiles=height(Td);
        fileQC=cell(0,23);
        for i=1:height(Td)
            nativeFs=NaN;nativeCount=NaN;frameCount=0;validCounts=zeros(1,4);voicedCount=0;
            status="Failed";detail="";cycleCounts=zeros(1,4);proxyCounts=zeros(1,4);
            try
                path=Td.FilePath(i);assert(isfile(path),'WAV not found.');
                info=audioinfo(path);assert(info.NumChannels==1,'Expected mono WAV; channels were not averaged.');
                assert(info.SampleRate==Td.NativeSampleRate(i),'WAV sample rate differs from native metadata.');
                assert(info.TotalSamples==Td.OriginalSampleCount(i),'WAV sample count differs from NSP metadata.');
                assert(info.BitsPerSample==16,'Expected exact native PCM16 WAV from the corrected converter.');
                [y,nativeFs]=audioread(path);nativeCount=numel(y);assert(all(isfinite(y)),'Nonfinite audio samples.');
                if cfg.VerifyNSPPayload
                    [original,originalFs]=vfReadNSP(Td.SourceNSP(i));
                    assert(originalFs==nativeFs && isequal(y,double(original)/32768),'WAV samples do not match the native NSP payload exactly.');
                end
                assert(nativeCount==Td.OriginalSampleCount(i),'Decoded sample count mismatch.');
                y=double(y(:));if nativeFs~=cfg.TargetFs,y=resample(y,cfg.TargetFs,nativeFs);end
                assert(~isempty(y),'Empty WAV.');y=y-mean(y);
                [frames,starts]=frameSignalNoPadding(y,n,hop);assert(size(frames,2)>=3,'Too few complete frames.');
                [F,active,voiced,confidence,fq]=extract43Features(frames,cfg.TargetFs,hamming(n,'periodic'), ...
                    cfg.NFFT,.85,bank,13,cfg.PitchFloor,cfg.PitchCeiling,.20,.70,.01,cfg.PerturbRadius);
                proxy=F(:,7:10);reference=vfMatlabCycleRecord(path,numel(y),cfg);
                [F,extra]=vfPerturbation(F,reference,window,voiced,cfg);
                cycleCounts=sum(extra.Sources=="Cycle",1);proxyCounts=sum(startsWith(extra.Sources,"Proxy_"),1);
                frameCount=size(F,1);assert(size(F,2)==43 && ~any(isinf(F),'all'));
                validCounts=sum(isfinite(F(:,7:10)),1);voicedCount=sum(voiced);
                FileName=repmat(Td.FileName(i),frameCount,1);SubjectID=repmat(Td.SubjectID(i),frameCount,1);
                Vowel=repmat(Td.Vowel(i),frameCount,1);Label=repmat(double(Td.Label(i)),frameCount,1);
                Gender=repmat(Td.Gender(i),frameCount,1);Disease=repmat(disease,frameCount,1);FrameID=(1:frameCount)';
                FrameStartTime=(starts(:)-1)/cfg.TargetFs;FrameCenterTime=((starts(:)-1)+(n-1)/2)/cfg.TargetFs;
                IsActive=active(:);IsVoiced=voiced(:);VoicingConfidence=confidence(:);
                base=table(FileName,SubjectID,Vowel,Label,Gender,Disease,FrameID,FrameStartTime,FrameCenterTime,IsActive,IsVoiced,VoicingConfidence);
                meta{i}=[base,vfDiagnosticTable(extra,proxy)];
                pure=F;pure(:,7:10)=extra.CycleValues;pureFeats{i}=pure;
                feats{i}=F;q.FilesProcessed=q.FilesProcessed+1;q.TotalFrames=q.TotalFrames+frameCount;
                q.ActiveFrames=q.ActiveFrames+sum(active);q.VoicedFrames=q.VoicedFrames+voicedCount;
                q.NaNFeatureValues=q.NaNFeatureValues+sum(isnan(F),'all');q.PitchOutliersRejected=q.PitchOutliersRejected+fq.PitchOutliersRejected;
                status="Completed";
            catch err
                q.ProcessingErrors=q.ProcessingErrors+1;detail=string(err.message);complete=false;
                if strcmp(getenv('GITHUB_ACTIONS'),'true')
                    report=getReport(err,'extended','hyperlinks','off');report=strrep(strrep(strrep(report,'%','%25'),char(13),'%0D'),char(10),'%0A');
                    fprintf(2,'::error title=Recording extraction failure::%s\n',report);
                end
            end
            fileQC(end+1,:)={disease,Td.SubjectID(i),Td.FileName(i),window,nativeFs,nativeCount, ...
                frameCount,voicedCount,validCounts(1),validCounts(2),validCounts(3),validCounts(4), ...
                cycleCounts(1),cycleCounts(2),cycleCounts(3),cycleCounts(4),proxyCounts(1),proxyCounts(2),proxyCounts(3),proxyCounts(4),status,detail,Td.FilePath(i)}; %#ok<AGROW>
        end
        files=cell2table(fileQC,'VariableNames',{'Disease','SubjectID','FileName','Window_ms','NativeFs','NativeSamples', ...
            'Frames','VoicedFrames','ValidJitterLocal','ValidJitterRAP','ValidShimmerLocal','ValidShimmerAPQ','CycleJitterLocal','CycleJitterRAP','CycleShimmerLocal','CycleShimmerAPQ5', ...
            'ProxyJitterLocal','ProxyJitterRAP','ProxyShimmerLocal','ProxyShimmerAPQ5','Status','Detail','FilePath'});
        q.CycleJitterLocal=sum(files.CycleJitterLocal);q.CycleJitterRAP=sum(files.CycleJitterRAP);
        q.CycleShimmerLocal=sum(files.CycleShimmerLocal);q.CycleShimmerAPQ5=sum(files.CycleShimmerAPQ5);
        q.ProxyJitterLocal=sum(files.ProxyJitterLocal);q.ProxyJitterRAP=sum(files.ProxyJitterRAP);
        q.ProxyShimmerLocal=sum(files.ProxyShimmerLocal);q.ProxyShimmerAPQ5=sum(files.ProxyShimmerAPQ5);
        keep=~cellfun(@isempty,feats);
        if any(keep)
            data=vertcat(feats{keep});metaTable=vertcat(meta{keep});Tout=[metaTable,array2table(data,'VariableNames',cellstr(settings.FeatureNames))];
            assert(height(Tout)<=1048575,'Excel row limit exceeded; no truncated workbook was written.');
            q.ActivePercent=100*q.ActiveFrames/max(q.TotalFrames,1);q.VoicedPercent=100*q.VoicedFrames/max(q.TotalFrames,1);
            q.OutputRows=height(Tout);q.FeaturesWithNoFiniteData=sum(sum(isfinite(data),1)==0);
            q.ValidJitterLocal=sum(isfinite(data(:,7)));q.ValidJitterRAP=sum(isfinite(data(:,8)));
            q.ValidShimmerLocal=sum(isfinite(data(:,9)));q.ValidShimmerAPQ=sum(isfinite(data(:,10)));
            file=fullfile(cfg.OutputFolder,sprintf('FullFeats_%dms_%s.xlsx',window,disease));
            vfAtomicTable(Tout,file);outputs(end+1,1)=file;
            if cfg.SaveCycleOnly
                cycleMeta=metaTable;sourceColumns=["JitterLocal_Source","JitterRAP_Source","ShimmerLocal_Source","ShimmerAPQ5_Source"];
                for sc=sourceColumns,v=cycleMeta.(sc);v(startsWith(v,"Proxy_") | v=="ProxyUnavailable")="InsufficientCycles_NoFallback";cycleMeta.(sc)=v;end
                cycleNames=settings.FeatureNames;cycleNames(7:10)=["JitterLocal","JitterRAP","ShimmerLocal","ShimmerAPQ5"];
                CycleOnly=[cycleMeta,array2table(vertcat(pureFeats{keep}),'VariableNames',cellstr(cycleNames))];
                pureFile=fullfile(cfg.OutputFolder,'CycleOnly',sprintf('FullFeats_%dms_%s.xlsx',window,disease));
                vfAtomicTable(CycleOnly,pureFile);outputs(end+1,1)=pureFile;
                if cfg.SaveMAT
                    pureSettings=settings;pureSettings.FeatureNames=cycleNames;pureSettings.PerturbationMode="cycle_only";pureSettings.ProxyFallbackWindowsMs=[];
                    vfSaveFeatures(replace(pureFile,'.xlsx','.mat'),CycleOnly,pureSettings);outputs(end+1,1)=replace(pureFile,'.xlsx','.mat');
                end
            end
            if cfg.SaveMAT,matfile=replace(file,'.xlsx','.mat');vfSaveFeatures(matfile,Tout,settings);outputs(end+1,1)=matfile;end
        else
            complete=false;q.ValidJitterLocal=0;q.ValidJitterRAP=0;q.ValidShimmerLocal=0;q.ValidShimmerAPQ=0;
        end
        groupQC=[groupQC;struct2table(q)];groupFiles=[groupFiles;files]; %#ok<AGROW>
        vfSaveQC([allQC;groupQC],[allFiles;groupFiles],cfg);
        fprintf('%s | %d ms | %d/%d recordings | %d frames | %d failures\n',disease,window,q.FilesProcessed,q.ExpectedFiles,q.TotalFrames,q.ProcessingErrors);
    end
    state=struct('Signature',signature,'Complete',complete,'Outputs',outputs,'QC',groupQC,'Files',groupFiles);
    vfAtomicSave(checkpoint,'state',state);
    allQC=[allQC;groupQC];allFiles=[allFiles;groupFiles]; %#ok<AGROW>
    vfSaveQC(allQC,allFiles,cfg);
end
status=struct('Complete',all(allQC.FilesProcessed==allQC.ExpectedFiles & allQC.ProcessingErrors==0), ...
    'ExpectedRecordingWindows',sum(allQC.ExpectedFiles),'ProcessedRecordingWindows',sum(allQC.FilesProcessed), ...
    'Failures',sum(allQC.ProcessingErrors),'ModelFeatureCount',43,'Mode',cfg.PerturbationMode);
vfWriteJSON(fullfile(cfg.OutputFolder,'Run_Status.json'),status);
assert(status.Complete,'Extraction finished with failures. Inspect File_Extraction_QC.xlsx; correct sources and rerun to resume.');
fprintf('Finished successfully. Inspect both QC workbooks before modeling.\n');
end

function cfg=vfConfiguration
cfg=struct('DatasetPath','D:\SVD\meta\svd_wav\data\Thesis_Final_Dataset.xlsx', ...
    'OutputFolder','D:\SVD\meta\svd_wav\Features_Final_NoPraat_HNRfix', ...
    'Diseases',["Healthy","Rekurrensparese","Hyperfunktionelle","Laryngitis",("Reinke "+string(char(214))+"dem"),"Spasmodische Dysphonie","Stimmlippenpolyp"], ...
    'TargetFs',16000,'WindowMs',[25,75],'HopMs',10,'NFFT',2048,'PerturbRadius',2, ...
    'PerturbationMode',"cycle_with_proxy_fallback",'ProxyFallbackWindowsMs',25, ...
    'EngineVersion',"MATLAB_YIN_AcousticCycles_v1",'EngineSHA256',"", ...
    'PitchFloor',50,'PitchCeiling',500,'MaxPeriodFactor',1.3,'MaxAmplitudeFactor',1.6, ...
    'AutoSelfTest',true,'VerifyNSPPayload',true,'SaveCycleOnly',true,'Resume',true,'SaveMAT',true,'Version',"Native_PCM_MATLAB_AcousticCycleOrProxy43_v1");
end


function q=initializeQC(winMs,diseaseName)
q=struct('Window_ms',winMs,'Disease',string(diseaseName),'FilesProcessed',0, ...
    'MissingFiles',0,'EmptyFiles',0,'TooShortFiles',0,'ProcessingErrors',0, ...
    'TotalFrames',0,'ActiveFrames',0,'ActivePercent',0,'VoicedFrames',0, ...
    'VoicedPercent',0,'NaNFeatureValues',0,'InfFeatureValues',0, ...
    'PitchOutliersRejected',0,'OutputRows',0,'FeaturesWithNoFiniteData',0);
end

function [subjectID,vowel]=parseSVDFileName(fileName)
subjectID=''; vowel='';
token=regexp(fileName,'_([^_]+)-([aiuAIU])(?:_|$)','tokens','once');
if ~isempty(token), subjectID=token{1}; vowel=lower(token{2}); return; end
token=regexp(fileName,'([0-9]+)-([aiuAIU])','tokens','once');
if ~isempty(token), subjectID=token{1}; vowel=lower(token{2}); return; end
error('Unparseable SVD filename: %s. No fallback SubjectID was assigned.',fileName);
end

function [frames,starts]=frameSignalNoPadding(y,winLen,hopLen)
N=length(y);
if N<winLen, frames=zeros(winLen,0); starts=zeros(1,0); return; end
starts=1:hopLen:(N-winLen+1); frames=zeros(winLen,numel(starts));
for k=1:numel(starts), idx=starts(k):(starts(k)+winLen-1); frames(:,k)=y(idx); end
end

function [F,activeMask,voicedMask,voiceConf,qc]=extract43Features(frames,fs,windowVector,nfft,rolloffPercent,melBank,numMFCC,f0Min,f0Max,yinThreshold,minVoicingConfidence,vadRelativeThreshold,perturbRadius)
[winLen,nFrames]=size(frames); windowVector=windowVector(:); qc.PitchOutliersRejected=0;

% Temporal base features
rmsVal=sqrt(mean(frames.^2,1)); peakAmp=max(abs(frames),[],1);
zcr=zeros(1,nFrames); entropyAmp=zeros(1,nFrames); energyEntropy=zeros(1,nFrames);
for f=1:nFrames
    x=frames(:,f); s=sign(x); s(s==0)=1; zcr(f)=sum(abs(diff(s))>0)/max(winLen-1,1);
    % Normalize by log2(N) so entropy scale is comparable across 25/75 ms.
    p=abs(x); p=p/(sum(p)+eps); entropyAmp(f)=-sum(p.*log2(p+eps))/log2(winLen);
    e=x.^2; e=e/(sum(e)+eps); energyEntropy(f)=-sum(e.*log2(e+eps))/log2(winLen);
end
zcrVar=localVariance(zcr,2);

% VAD
positiveRms=rmsVal(rmsVal>0 & isfinite(rmsVal));
if isempty(positiveRms), rmsThreshold=0; else, rmsThreshold=max(vadRelativeThreshold*max(positiveRms),0.10*median(positiveRms)); end
activeMask=rmsVal>rmsThreshold;

% YIN F0
f0=nan(1,nFrames); voiceConf=zeros(1,nFrames);
for f=1:nFrames
    if activeMask(f), [f0(f),voiceConf(f)]=estimateF0YIN(frames(:,f),fs,f0Min,f0Max,yinThreshold); end
end
rawVoiced=activeMask & isfinite(f0) & f0>=f0Min & f0<=f0Max & voiceConf>=minVoicingConfidence;
[f0,voicedMask,nRejected]=cleanPitchTrack(f0,rawVoiced); qc.PitchOutliersRejected=nRejected;

% Jitter/Shimmer frame-based perturbation
jitterLocal=nan(1,nFrames); jitterRAP=nan(1,nFrames); shimmerLocal=nan(1,nFrames); shimmerAPQ=nan(1,nFrames);
period=nan(1,nFrames); period(voicedMask)=1./f0(voicedMask); amplitude=rmsVal;
for f=1:nFrames
    if ~voicedMask(f), continue; end
    lo=max(1,f-perturbRadius); hi=min(nFrames,f+perturbRadius);
    left=f;right=f;
    while left>lo && voicedMask(left-1),left=left-1;end
    while right<hi && voicedMask(right+1),right=right+1;end
    idx=left:right;
    T=period(idx); A=amplitude(idx); T=T(isfinite(T)&T>0); A=A(isfinite(A)&A>eps);
    if numel(T)>=2, jitterLocal(f)=100*mean(abs(diff(T)))/(mean(T)+eps); end
    if numel(T)>=3
        rapTerms=zeros(numel(T)-2,1);
        for j=2:numel(T)-1, rapTerms(j-1)=abs(T(j)-mean(T(j-1:j+1))); end
        jitterRAP(f)=100*mean(rapTerms)/(mean(T)+eps);
    end
    if numel(A)>=2, shimmerLocal(f)=100*mean(abs(diff(A)))/(mean(A)+eps); end
    if numel(A)>=5
        apqTerms=zeros(numel(A)-4,1);
        for j=3:numel(A)-2, apqTerms(j-2)=abs(A(j)-mean(A(j-2:j+2))); end
        shimmerAPQ(f)=100*mean(apqTerms)/(mean(A)+eps);
    end
end

% Spectrum
nBins=floor(nfft/2)+1; powerSpec=zeros(nBins,nFrames); magnitudeNorm=zeros(nBins,nFrames);
for f=1:nFrames
    xw=frames(:,f).*windowVector; X=fft(xw,nfft); mag=abs(X(1:nBins)); pwr=mag.^2; powerSpec(:,f)=pwr; magnitudeNorm(:,f)=mag/(sqrt(sum(mag.^2))+eps);
end
freqAxis=(0:nBins-1)'*fs/nfft;

specCentroid=zeros(1,nFrames); specRolloff=zeros(1,nFrames); specFlux=zeros(1,nFrames); specEntropy=zeros(1,nFrames);
for f=1:nFrames
    pwr=powerSpec(:,f); p=pwr/(sum(pwr)+eps); specCentroid(f)=sum(freqAxis.*p);
    c=cumsum(pwr); idx=find(c>=rolloffPercent*(sum(pwr)+eps),1,'first'); if isempty(idx), idx=nBins; end; specRolloff(f)=freqAxis(idx);
    specEntropy(f)=-sum(p.*log2(p+eps));
    if f>1, ds=magnitudeNorm(:,f)-magnitudeNorm(:,f-1); specFlux(f)=sqrt(sum(ds.^2)); else, specFlux(f)=0; end
end

% HNR autocorrelation based
hnr=nan(1,nFrames);
for f=1:nFrames, if voicedMask(f), hnr(f)=estimateHNRAutocorr(frames(:,f),fs,f0(f),windowVector); end, end

% MFCC from the exact same frames
mfccMat=zeros(nFrames,numMFCC);
for f=1:nFrames
    melEnergy=melBank*powerSpec(:,f); logMel=log(melEnergy+eps); c=dct(logMel); mfccMat(f,:)=c(1:numMFCC).';
end
deltaMFCC=computeDelta(mfccMat,2);

% Formants
F1=nan(1,nFrames); F2=nan(1,nFrames);
for f=1:nFrames, if voicedMask(f), [F1(f),F2(f)]=estimateFormantsLPC(frames(:,f).*windowVector,fs); end, end

F=[zcr(:),zcrVar(:),rmsVal(:),peakAmp(:),entropyAmp(:),energyEntropy(:),jitterLocal(:),jitterRAP(:),shimmerLocal(:),shimmerAPQ(:),hnr(:),specCentroid(:),specRolloff(:),specFlux(:),specEntropy(:),mfccMat,deltaMFCC,F1(:),F2(:)];
if size(F,2)~=43, error('Expected 43 features, got %d',size(F,2)); end
F(isinf(F))=NaN;
end

function v=localVariance(x,radius)
n=numel(x); v=zeros(size(x));
for i=1:n
    vals=x(max(1,i-radius):min(n,i+radius)); vals=vals(isfinite(vals));
    if numel(vals)>1, v(i)=var(vals,0); else, v(i)=0; end
end
end

function [f0,confidence]=estimateF0YIN(x,fs,f0Min,f0Max,threshold)
x=double(x(:)); x=x-mean(x); N=length(x); tauMin=max(2,floor(fs/f0Max)); tauMax=min(floor(fs/f0Min),N-2); f0=NaN; confidence=0;
if tauMax<=tauMin+2 || sqrt(mean(x.^2))<eps, return; end
d=zeros(tauMax,1);
for tau=1:tauMax, nv=N-tau; dv=x(1:nv)-x(1+tau:N); d(tau)=sum(dv.^2); end
cmnd=yinCumulativeMean(d);
tauCandidate=NaN;
for tau=tauMin:tauMax-1
    if cmnd(tau)<threshold
        while tau<tauMax-1 && cmnd(tau+1)<cmnd(tau), tau=tau+1; end
        tauCandidate=tau; break;
    end
end
if isnan(tauCandidate)
    [minVal,rel]=min(cmnd(tauMin:tauMax)); tauCandidate=tauMin+rel-1; if minVal>0.45, return; end
end
refinedTau=tauCandidate;
if tauCandidate>1 && tauCandidate<tauMax
    y1=cmnd(tauCandidate-1); y2=cmnd(tauCandidate); y3=cmnd(tauCandidate+1); den=y1-2*y2+y3;
    if abs(den)>eps, refinedTau=tauCandidate+0.5*(y1-y3)/den; end
end
candidateF0=fs/refinedTau; if candidateF0<f0Min || candidateF0>f0Max, return; end
f0=candidateF0; confidence=max(0,min(1,1-cmnd(tauCandidate)));
end

function [cleanF0,cleanMask,rejected]=cleanPitchTrack(f0,voicedMask)
cleanF0=f0; cleanMask=voicedMask; rejected=0; validIdx=find(voicedMask); if numel(validIdx)<3, return; end
for ii=1:numel(validIdx)
    f=validIdx(ii); lo=max(1,f-2); hi=min(numel(f0),f+2); nb=f0(lo:hi); mk=voicedMask(lo:hi); nb=nb(mk & isfinite(nb));
    if numel(nb)>=3
        medF0=median(nb); if medF0>0 && abs(f0(f)-medF0)/medF0>0.40, cleanMask(f)=false; cleanF0(f)=NaN; rejected=rejected+1; end
    end
end
end

function hnrDb=estimateHNRAutocorr(x,fs,f0,w)
% Window-weighted normalized correlation of overlapping RAW samples.
% Fractional-period sinc refinement avoids integer-lag quantization bias.
% This is a finite-frame HNR estimate, not Praat Harmonicity.
hnrDb=NaN;
if ~isscalar(f0)||~isfinite(f0)||f0<=0,return;end
x=double(x(:));w=double(w(:));n=numel(x);
if numel(w)~=n||any(~isfinite(x))||any(~isfinite(w))||any(w<0)||~any(w>0),return;end
lag0=round(fs/f0);if lag0<1||n<2*lag0,return;end
x=x-mean(x);scale=max(abs(x));if scale<=realmin,return;end
x=x/scale;w=w/max(w);radius=max(1,round(.15*lag0));
lo=max(1,lag0-radius);hi=min(n-34,lag0+radius);if lo>hi,return;end
lags=lo:hi;values=arrayfun(@(lag)vfHNRLag(x,w,lag),lags);
[peak,index]=max(values);left=max(lo,lags(index)-1);right=min(hi,lags(index)+1);
if left<right
    [~,negativePeak,flag]=fminbnd(@(lag)-vfHNRLag(x,w,lag),left,right, ...
        optimset('Display','off','TolX',1e-5,'MaxIter',50));
    if flag<=0,return;end
    peak=max(peak,-negativePeak);
end
if ~isfinite(peak)||peak<=0,return;end
% Numerical bound only: no clinical threshold or window-dependent offset.
peak=min(max(peak,1e-12),1-1e-12);hnrDb=10*log10(peak/(1-peak));
end

function r=vfHNRLag(x,w,tau)
half=32;lag=floor(tau+.5);delta=tau-lag;n=numel(x);
first=max(0,half-lag);last=n-half-lag-1;r=-1;
if last-first+1<8,return;end
indices=(first:last)';shifted=indices+lag;a=x(indices+1);
if abs(delta)<1e-12
    b=x(shifted+1);
else
    taps=(-half:half)';kernel=sinc(delta-taps).*kaiser(2*half+1,8.6);
    kernel=kernel/sum(kernel);filtered=conv(x,flipud(kernel),'valid');
    b=filtered(shifted-half+1);
end
weights=w(indices+1).*interp1((0:n-1)',w,shifted+delta,'linear');
denominator=sqrt(sum(weights.*a.^2)*sum(weights.*b.^2));
if denominator>realmin,r=sum(weights.*a.*b)/denominator;end
end

function melBank=createMelFilterBank(fs,nfft,nFilters,fMin,fMax)
nBins=floor(nfft/2)+1; hz2mel=@(hz)2595*log10(1+hz/700); mel2hz=@(mel)700*(10.^(mel/2595)-1);
melPoints=linspace(hz2mel(fMin),hz2mel(fMax),nFilters+2); hzPoints=mel2hz(melPoints); bins=floor((nfft+1)*hzPoints/fs); bins=max(0,min(nBins-1,bins)); melBank=zeros(nFilters,nBins);
for m=1:nFilters
    left=bins(m); center=bins(m+1); right=bins(m+2); if center<=left, center=left+1; end; if right<=center, right=center+1; end; center=min(center,nBins-1); right=min(right,nBins-1);
    for k=left:center, melBank(m,k+1)=(k-left)/max(center-left,1); end
    for k=center:right, melBank(m,k+1)=(right-k)/max(right-center,1); end
end
rs=sum(melBank,2); rs(rs==0)=1; melBank=melBank./rs;
end

function delta=computeDelta(C,N)
[nFrames,nCoeff]=size(C); delta=zeros(nFrames,nCoeff); denom=2*sum((1:N).^2);
for t=1:nFrames
    num=zeros(1,nCoeff);
    for n=1:N, tp=min(nFrames,t+n); tm=max(1,t-n); num=num+n*(C(tp,:)-C(tm,:)); end
    delta(t,:)=num/denom;
end
end

function [F1,F2]=estimateFormantsLPC(x,fs)
F1=NaN; F2=NaN; x=double(x(:)); x=x-mean(x); if length(x)<100 || sqrt(mean(x.^2))<eps, return; end
x=filter([1 -0.97],1,x); order=round(fs/1000)+2; order=min(order,floor(length(x)/4)); if order<8, return; end
try
    a=lpc(x,order); r=roots(a); r=r(imag(r)>0); [freqs,bws]=poleFrequencyBandwidth(r,fs); valid=freqs>150 & freqs<4000 & bws>0 & bws<700; freqs=sort(freqs(valid));
    if numel(freqs)>=1, F1=freqs(1); end; if numel(freqs)>=2, F2=freqs(2); end
    if isfinite(F1)&&(F1<150||F1>1500), F1=NaN; end; if isfinite(F2)&&(F2<500||F2>4000), F2=NaN; end; if isfinite(F1)&&isfinite(F2)&&F2<=F1, F1=NaN; F2=NaN; end
catch, F1=NaN; F2=NaN; end
end


function cmnd=yinCumulativeMean(d)
% Standard cumulative mean normalized difference: sum includes d(1).
d=double(d(:)); cmnd=ones(size(d)); cumulative=cumsum(d);
for tau=2:numel(d)
    if cumulative(tau)>eps
        cmnd(tau)=tau*d(tau)/cumulative(tau);
    end
end
end

function [freqs,bws]=poleFrequencyBandwidth(r,fs)
% For r=exp(-pi*BW/fs)*exp(1i*2*pi*f/fs), BW is in Hz.
freqs=atan2(imag(r),real(r))*fs/(2*pi);
bws=-(fs/pi).*log(abs(r));
end

function extractorSelfTest()
% Numerical oracles, not a research-performance test. No patient data used.
rng(1405,'twister'); fs=16000;
% A known difference sequence exposes omission of its first term.
d=[3;7;2;5;11]; c=yinCumulativeMean(d);
expected=[1;14/10;6/12;20/17;55/28];
assert(max(abs(c-expected))<1e-12,'YIN normalization regression.');
assert(all(yinCumulativeMean(zeros(8,1))==1),'Zero difference must not create spurious confidence.');
for ms=[25,75]
    n=round(ms*fs/1000); t=(0:n-1)'/fs;
    for frequency=[80,160,260,400]
        x=sin(2*pi*frequency*t);
        [f0,confidence]=estimateF0YIN(x,fs,50,500,.20);
        assert(isfinite(f0) && abs(f0-frequency)/frequency<.03, ...
            'YIN synthetic F0 outside 3 percent tolerance.');
        assert(confidence>=.70 && confidence<=1);
    end
    [f0,confidence]=estimateF0YIN(zeros(n,1),fs,50,500,.20);
    assert(isnan(f0) && confidence==0,'Silence must be unvoiced.');
end
% Pole radii synthesized from known physical bandwidths recover those Hz.
f=[500;1500;2500]; bw=[80;120;200];
r=exp(-pi*bw/fs).*exp(1i*2*pi*f/fs);
[estimatedF,estimatedBW]=poleFrequencyBandwidth(r,fs);
assert(max(abs(estimatedF-f))<1e-8 && max(abs(estimatedBW-bw))<1e-8, ...
    'LPC frequency/bandwidth conversion regression.');
% Physical controls: noiseless periodic signals at integer/fractional lags.
for frequency=[80,93.7,200,237.3,400]
    for duration=[25,75]
        count=round(fs*duration/1000);tt=(0:count-1)'/fs;ww=hamming(count,'periodic');
        cleanSignal=sin(2*pi*frequency*tt+.37);
        value=estimateHNRAutocorr(cleanSignal,fs,frequency,ww);
        assert(value>90,'Pure periodic HNR regression.');
    end
end
assert(isnan(estimateHNRAutocorr(zeros(400,1),fs,200,hamming(400,'periodic'))));
assert(isnan(estimateHNRAutocorr(ones(400,1),fs,200,hamming(400,'periodic'))));
assert(isnan(estimateHNRAutocorr(randn(400,1),fs,50,hamming(400,'periodic'))));
% Known additive noise controls, independent of any patient/class labels.
for targetSNR=[0,10,20,30]
    for duration=[25,75]
        count=round(fs*duration/1000);tt=(0:count-1)'/fs;ww=hamming(count,'periodic');
        estimates=zeros(24,1);
        for trial=1:24
            periodic=sin(2*pi*237.3*tt+2*pi*rand);noise=randn(count,1);noise=noise-mean(noise);
            noise=noise*sqrt(mean(periodic.^2)/(10^(targetSNR/10)*mean(noise.^2)));
            estimates(trial)=estimateHNRAutocorr(periodic+noise,fs,237.3,ww);
        end
        assert(abs(median(estimates)-targetSNR)<2,'Known-noise HNR calibration regression.');
    end
end
% Exercise the entire feature kernel on both base windows, including NaNs
% legitimately produced by unvoiced/unresolved perturbation/formant estimates.
t=(0:fs-1)'/fs; signal=.5*sin(2*pi*160*t)+.2*sin(2*pi*320*t)+.02*randn(size(t));
counts=zeros(1,2);
for k=1:2
    ms=25; if k==2, ms=75; end
    win=round(ms*fs/1000); [frames,starts]=frameSignalNoPadding(signal,win,160);
    bank=createMelFilterBank(fs,2048,26,0,fs/2);
    [F,active,voiced,confidence,qc]=extract43Features(frames,fs,hamming(win,'periodic'), ...
        2048,.85,bank,13,50,500,.20,.70,.01,2);
    assert(size(F,2)==43 && size(F,1)==numel(starts));
    assert(~any(isinf(F),'all') && any(voiced) && all(isfinite(confidence)));
    assert(numel(active)==size(F,1) && qc.PitchOutliersRejected>=0);
    assert(all(abs(diff((starts-1)/fs)-.01)<1e-12));
    counts(k)=size(F,1);
end
assert(counts(1)-counts(2)==5);
[id,vowel]=parseSVDFileName('Healthy_1011-a_n');
assert(strcmp(id,'1011') && strcmp(vowel,'a'));
rejected=false;
try
    parseSVDFileName('unparseable_recording');
catch
    rejected=true;
end
assert(rejected,'Unparseable identity must not silently become a subject.');
fprintf('EXTRACTOR_NUMERIC_TESTS_PASSED\n');
end

function [T,audit]=vfValidateMetadata(T,cfg)
required={'FilePath','Label','Gender','Disease','SubjectID','Vowel','SourceNSP','NativeSampleRate','OriginalSampleCount'};
assert(all(ismember(required,T.Properties.VariableNames)), ...
    'Use the NEW Thesis_Final_Dataset.xlsx from NSP_to_WAV_Corrected; native source metadata columns are required.');
for name=["FilePath","SourceNSP"]
    v=string(T.(name));for j=1:numel(v),v(j)=vfNormalizePath(v(j));end;T.(name)=v;
end
for name=["Gender","Disease","SubjectID","Vowel"],T.(name)=strip(string(T.(name)));end
T.Gender=lower(T.Gender);T.Vowel=lower(T.Vowel);
assert(isnumeric(T.Label)&&all(ismember(T.Label,[0,1])),'Labels must be numeric 0/1.');
assert(all(ismember(T.Gender,["m","w"])),'Gender must be m/w as in the inspected CSV.');
assert(all(ismember(T.Vowel,["a","i","u"])),'Expected a/i/u vowels.');
assert(isnumeric(T.NativeSampleRate)&&all(isfinite(T.NativeSampleRate)&T.NativeSampleRate>=1000));
assert(isnumeric(T.OriginalSampleCount)&&all(isfinite(T.OriginalSampleCount)&T.OriginalSampleCount>0));
assert(all(mod(T.NativeSampleRate,1)==0) && all(mod(T.OriginalSampleCount,1)==0));
assert(all(~ismissing(T.FilePath)&strlength(T.FilePath)>0));
assert(all(~ismissing(T.SubjectID)&strlength(T.SubjectID)>0));
[~,first,g]=unique(lower(T.FilePath),'stable');
for j=1:numel(first)
    rows=find(g==j);
    if numel(rows)>1
        for r=rows(2:end)'
            assert(isequaln(T(rows(1),required),T(r,required)),'Same WAV has conflicting metadata.');
        end
    end
end
T=T(first,:);T=T(ismember(T.Disease,cfg.Diseases),:);assert(~isempty(T),'No selected diseases found.');
healthy=T.Disease=="Healthy";
assert(all(T.Label(healthy)==0)&&all(T.Label(~healthy)==1),'Disease/label conflict.');
assert(isempty(intersect(T.SubjectID(healthy),T.SubjectID(~healthy))),'A subject is both Healthy and Patient.');
T.FileName=strings(height(T),1);
for j=1:height(T)
    [~,name,ext]=fileparts(T.FilePath(j));assert(strcmpi(ext,'.wav'),'Expected converted WAV files.');
    [id,vowel]=parseSVDFileName(char(name));
    assert(string(id)==T.SubjectID(j)&&string(vowel)==T.Vowel(j),'Filename/metadata identity mismatch.');
    assert(startsWith(name,T.Disease(j)+"_"),'Disease-specific output filename mismatch.');
    [~,source]=fileparts(T.SourceNSP(j));[sourceID,sourceVowel]=parseSVDFileName(char(source));
    assert(string(sourceID)==T.SubjectID(j)&&string(sourceVowel)==T.Vowel(j),'NSP/metadata identity mismatch.');
    T.FileName(j)=name;
end
key=T.Disease+"|"+T.SubjectID+"|"+T.Vowel;
assert(numel(unique(key))==height(T),'More than one source for a disease/subject/vowel. No arbitrary recording was selected.');
ids=unique(T.SubjectID);multiple=strings(0,1);
for id=ids'
    rows=T.SubjectID==id;assert(numel(unique(T.Gender(rows)))==1,'Subject gender conflict.');
    if numel(unique(T.Disease(rows)))>1,multiple(end+1,1)=id;end %#ok<AGROW>
end
audit=table();
for disease=cfg.Diseases
    q=T(T.Disease==disease,:);if isempty(q),continue;end
    a=table(disease,height(q),numel(unique(q.SubjectID)),numel(intersect(unique(q.SubjectID),multiple)), ...
        'VariableNames',{'Disease','Recordings','Subjects','MultipleDiseaseSubjects'});
    audit=[audit;a]; %#ok<AGROW>
end
end

function path=vfNormalizePath(path)
path=replace(string(path),'\','/');unc=startsWith(path,'//');path=regexprep(path,'/+','/');
if unc,path="/"+path;end
end

function names=vfFeatureNames(cfg)
names=["ZCR","ZCR_Var","RMS","PeakAmp","Entropy","EnergyEntropy", ...
    "JitterProxy_local","JitterProxy_rap","ShimmerProxy_local","ShimmerProxy_apq", ...
    "HNR","SpecCentroid","SpecRolloff","SpecFlux","SpecEntropy", ...
    "MFCC"+string(1:13),"DeltaMFCC"+string(1:13),"FormantF1","FormantF2"];
names(7:10)=["JitterLocal_CycleOrProxy","JitterRAP_CycleOrProxy","ShimmerLocal_CycleOrProxy","ShimmerAPQ5_CycleOrProxy"];
assert(numel(names)==43);
end

function signature=vfGroupSignature(T,settings)
settings.Resume=false;blob=string(jsonencode(settings))+string(newline);
for j=1:height(T)
    p=T.FilePath(j);d=dir(p);
    if isempty(d),fileInfo="MISSING";else,fileInfo=vfHashFile(p);end
    blob=blob+strjoin([p,string(T.Label(j)),T.Gender(j),T.Disease(j),T.SubjectID(j),T.Vowel(j), ...
        string(T.NativeSampleRate(j)),string(T.OriginalSampleCount(j)),fileInfo],"|")+newline;
end
assert(isscalar(blob),'Group signature serialization must be scalar text.');signature=vfHashText(blob);
end

function h=vfHashFile(file)
f=fopen(file,'rb');assert(f>=0);c=onCleanup(@()fclose(f)); %#ok<NASGU>
md=java.security.MessageDigest.getInstance('SHA-256');
while true,b=fread(f,1048576,'*uint8');if isempty(b),break;end;md.update(typecast(b,'int8'));end
h=vfDigest(md);
end

function h=vfDigest(md)
h=string(lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[])));
end

function vfAtomicTable(t,file)
folder=fileparts(file);tmp=[tempname(folder) '.xlsx'];writetable(t,tmp);movefile(tmp,file,'f');
end

function vfAtomicSave(file,name,value)
p=struct();p.(name)=value;tmp=[tempname(fileparts(file)) '.mat'];save(tmp,'-struct','p','-v7.3');movefile(tmp,file,'f');
end

function vfWriteJSON(file,value)
f=fopen(file,'w','n','UTF-8');assert(f>=0);c=onCleanup(@()fclose(f));fprintf(f,'%s',jsonencode(value)); %#ok<NASGU>
end

function vfSaveQC(q,files,cfg)
vfAtomicTable(q,fullfile(cfg.OutputFolder,'Feature_Extraction_QC.xlsx'));
vfAtomicTable(files,fullfile(cfg.OutputFolder,'File_Extraction_QC.xlsx'));
end

function vfInputSelfTest
cfg=vfConfiguration();T=table(["/tmp/Healthy_1-a_n.wav";"/tmp/Rekurrensparese_106-a_n.wav";("/tmp/Reinke "+string(char(214))+"dem_106-a_n.wav")], ...
    [0;1;1],["w";"m";"m"],["Healthy";"Rekurrensparese";("Reinke "+string(char(214))+"dem")],["1";"106";"106"],["a";"a";"a"], ...
    ["/tmp/h/1-a_n.nsp";"/tmp/r/106-a_n.nsp";"/tmp/e/106-a_n.nsp"],repmat(50000,3,1),repmat(50000,3,1), ...
    'VariableNames',{'FilePath','Label','Gender','Disease','SubjectID','Vowel','SourceNSP','NativeSampleRate','OriginalSampleCount'});
[clean,audit]=vfValidateMetadata(T,cfg);assert(height(clean)==3&&sum(audit.MultipleDiseaseSubjects)==2);
D=[T;T(1,:)];[clean,~]=vfValidateMetadata(D,cfg);assert(height(clean)==3);
D=T;D.SubjectID(2)="1";failed=false;try,vfValidateMetadata(D,cfg);catch,failed=true;end;assert(failed);
D=T;D.Label(2)=0;failed=false;try,vfValidateMetadata(D,cfg);catch,failed=true;end;assert(failed);
fprintf('METADATA_IDENTITY_AND_MULTIPLE_MEMBERSHIP_CHECKS_PASSED\n');
end


function h=vfHashText(text)
md=java.security.MessageDigest.getInstance('SHA-256');b=unicode2native(char(text),'UTF-8');md.update(typecast(uint8(b),'int8'));h=vfDigest(md);
end

function reference=vfMatlabCycleRecord(wav,nResampled,cfg)
% This backend estimates acoustic cycle landmarks, not physical glottal closures.
cache=fullfile(cfg.OutputFolder,'_cycle_cache');if ~isfolder(cache),mkdir(cache);end
counts=max(0,floor((nResampled-round(cfg.WindowMs*cfg.TargetFs/1000))/round(cfg.HopMs*cfg.TargetFs/1000))+1);
audioHash=vfHashFile(wav);codeHash=vfHashFile(string(mfilename('fullpath'))+".m");
key=vfHashText(audioHash+codeHash+jsonencode({cfg.EngineVersion,cfg.WindowMs,cfg.HopMs,cfg.PitchFloor,cfg.PitchCeiling,cfg.MaxPeriodFactor,cfg.MaxAmplitudeFactor}));
file=fullfile(cache,key+".mat");
if isfile(file)
    saved=load(file,'reference');reference=saved.reference;
    assert(reference.Signature==key&&isequal(reference.FrameCounts,counts),'Cycle cache mismatch.');return;
end
[x,fs]=audioread(wav);assert(size(x,2)==1&&all(isfinite(x)),'Invalid mono input.');x=x-mean(x);
[pulses,trackTimes,trackF0]=vfAcousticPulses(x,fs,cfg);
rows=sum(counts);Window_ms=zeros(rows,1);FrameID=zeros(rows,1);metrics=nan(rows,4);CycleF0=nan(rows,1);offset=0;
for windowIndex=1:numel(cfg.WindowMs)
    window=cfg.WindowMs(windowIndex);
    for i=1:counts(windowIndex)
        row=offset+i;t0=(i-1)*cfg.HopMs/1000;t1=t0+window/1000;
        Window_ms(row)=window;FrameID(row)=i;
        local=pulses(pulses>=t0 & pulses<=t1);
        periods=diff(local);amplitudes=nan(max(0,numel(local)-2),1);
        % Interior cycles use midpoint-to-midpoint support bounded by the frame.
        for j=2:numel(local)-1
            left=max(1,ceil((local(j-1)+local(j))*fs/2)+1);
            right=min(numel(x),floor((local(j)+local(j+1))*fs/2)+1);
            if right>left,amplitudes(j-1)=max(x(left:right))-min(x(left:right));end
        end
        metrics(row,:)=vfAcousticMetrics(periods,amplitudes,cfg);
        if ~isempty(trackTimes)
            [~,near]=min(abs(trackTimes-(t0+t1)/2));CycleF0(row)=trackF0(near);
        end
    end
    offset=offset+counts(windowIndex);
end
R=table(Window_ms,FrameID,metrics(:,1),metrics(:,2),metrics(:,3),metrics(:,4),CycleF0, ...
    'VariableNames',{'Window_ms','FrameID','JitterLocal','JitterRAP','ShimmerLocal','ShimmerAPQ5','CycleF0'});
reference=struct('Signature',key,'FrameCounts',counts,'Metrics',R,'Pulses',pulses, ...
    'AudioSHA256',audioHash,'EngineVersion',cfg.EngineVersion);
vfAtomicSave(file,'reference',reference);
end

function [points,times,f0]=vfAcousticPulses(x,fs,cfg)
% YIN is estimated in 60ms context; landmarks remain sample-based at native fs.
% Only stable voiced runs are tracked. Ambiguous/unvoiced cycles remain missing.
y=x;if fs~=cfg.TargetFs,y=resample(y,cfg.TargetFs,fs);end
step=round(cfg.HopMs*cfg.TargetFs/1000);context=round(.060*cfg.TargetFs);
centres=(1:step:numel(y))';times=(centres-1)/cfg.TargetFs;f0=nan(size(times));
energy=sqrt(mean(y.^2));points=zeros(0,1);
if isempty(y)||~isfinite(energy)||energy<=eps,return;end
for i=1:numel(centres)
    left=max(1,centres(i)-floor(context/2));right=min(numel(y),left+context-1);left=max(1,right-context+1);
    frame=y(left:right);
    if sqrt(mean(frame.^2))<max(1e-7,.05*energy),continue;end
    [pitch,confidence]=estimateF0YIN(frame,cfg.TargetFs,cfg.PitchFloor,cfg.PitchCeiling,.20);
    if confidence>=.70,f0(i)=pitch;end
end
[f0,voiced]=cleanPitchTrack(f0,isfinite(f0));f0(~voiced)=NaN;
% Remove only high-frequency tracking noise; amplitudes use unfiltered native x.
cutoff=min(4000,.45*fs);[b,a]=butter(3,cutoff/(fs/2));
if numel(x)<=3*max(numel(a),numel(b)),return;end
tracking=filtfilt(b,a,x);runStart=find(voiced & [true;~voiced(1:end-1)]);runEnd=find(voiced & [~voiced(2:end);true]);
for run=1:numel(runStart)
    first=runStart(run);last=runEnd(run);
    begin=max(1,round(max(0,times(first)-cfg.HopMs/2000)*fs)+1);
    finish=min(numel(x)-1,round((times(last)+cfg.HopMs/2000)*fs)+1);
    period=fs/f0(first);seedEnd=min(finish,begin+ceil(1.5*period));
    if seedEnd<=begin+2,continue;end
    [~,relative]=max(tracking(begin:seedEnd));position=begin+relative-1;
    while position<finish
        fractional=vfRefinedPeak(tracking,position);points(end+1,1)=(fractional-1)/fs; %#ok<AGROW>
        nearest=min(last,max(first,round(((position-1)/fs)/ (cfg.HopMs/1000))+1));
        period=fs/f0(nearest);
        low=max(position+1,ceil(position+.70*period));high=min(finish,floor(position+1.30*period));
        if high<=low+2,break;end
        segment=tracking(low:high);[~,candidates]=findpeaks(segment);
        if isempty(candidates),break;end
        candidates=candidates+low-1;
        % Prefer the strongest consistent positive peak in the predicted cycle.
        [~,best]=max(tracking(candidates));next=candidates(best);
        if tracking(next)<=0,break;end
        position=next;
    end
end
points=unique(points);points=points(points>=0 & points<=(numel(x)-1)/fs);
end

function position=vfRefinedPeak(x,index)
position=double(index);
if index>1&&index<numel(x)
    denominator=x(index-1)-2*x(index)+x(index+1);
    if denominator<0
        delta=.5*(x(index-1)-x(index+1))/denominator;
        if abs(delta)<=.5,position=position+delta;end
    end
end
end

function metrics=vfAcousticMetrics(periods,amplitudes,cfg)
% Percent units. No compression over rejected cycles or amplitude ratios.
metrics=nan(1,4);periods=periods(:);amplitudes=amplitudes(:);
if isempty(periods),return;end
valid=isfinite(periods)&periods>=1/cfg.PitchCeiling&periods<=1/cfg.PitchFloor;
ratio=max(periods(1:end-1)./periods(2:end),periods(2:end)./periods(1:end-1));
% Conservative rule: any rejected interval invalidates this frame's cycle metrics.
if ~all(valid)||any(ratio>cfg.MaxPeriodFactor),return;end
meanPeriod=mean(periods);
if numel(periods)>=2,metrics(1)=100*mean(abs(diff(periods)))/meanPeriod;end
if numel(periods)>=3
    local=(periods(1:end-2)+periods(2:end-1)+periods(3:end))/3;
    metrics(2)=100*mean(abs(periods(2:end-1)-local))/meanPeriod;
end
if numel(amplitudes)<2||any(~isfinite(amplitudes)|amplitudes<=0),return;end
amplitudeRatio=max(amplitudes(1:end-1)./amplitudes(2:end),amplitudes(2:end)./amplitudes(1:end-1));
if any(amplitudeRatio>cfg.MaxAmplitudeFactor),return;end
metrics(3)=100*mean(abs(diff(amplitudes)))/mean(amplitudes);
if numel(amplitudes)>=5
    smoothed=conv(amplitudes,ones(5,1)/5,'valid');
    metrics(4)=100*mean(abs(amplitudes(3:end-2)-smoothed))/mean(amplitudes);
end
end

function [F,extra]=vfPerturbation(F,reference,window,voiced,cfg)
R=reference.Metrics(reference.Metrics.Window_ms==window,:);assert(height(R)==size(F,1));
cycle=R{:,{'JitterLocal','JitterRAP','ShimmerLocal','ShimmerAPQ5'}};proxy=F(:,7:10);
pulses=reference.Pulses;intervals=diff(pulses);bounds=[1/cfg.PitchCeiling,1/cfg.PitchFloor];
n=size(F,1);rawCounts=zeros(n,1);validRuns=rawCounts;periodic=false(n,1);invalid=false(n,1);
for i=1:n
    t0=(i-1)*cfg.HopMs/1000;t1=t0+window/1000;
    local=pulses(pulses>=t0&pulses<=t1);d=diff(local);rawCounts(i)=numel(d);
    valid=d>=bounds(1)&d<=bounds(2);ratio=max(d(1:end-1)./d(2:end),d(2:end)./d(1:end-1));
    invalid(i)=any(~valid)||any(ratio>cfg.MaxPeriodFactor);
    run=0;best=0;
    for j=1:numel(d)
        if ~valid(j),run=0;
        elseif j>1 && valid(j-1) && ratio(j-1)<=cfg.MaxPeriodFactor,run=run+1;
        else,run=1;end
        best=max(best,run);
    end
    validRuns(i)=best;
    overlap=pulses(1:end-1)<=t1 & pulses(2:end)>=t0;
    periodic(i)=any(overlap & intervals>=bounds(1) & intervals<=bounds(2));
end
[mixed,source]=vfCombineCycleProxy(cycle,proxy,rawCounts,periodic,invalid,voiced(:),window,cfg);
F(:,7:10)=mixed;
extra=struct('CycleValues',cycle,'Sources',source,'CompleteCycles',rawCounts, ...
    'LongestValidCycleRun',validRuns,'CycleF0',R.CycleF0);
end

function [mixed,source]=vfCombineCycleProxy(cycle,proxy,counts,support,invalid,voiced,window,cfg)
% Interior acoustic amplitude estimates need bordering pulses: shimmer local requires
% >=3 PointProcess intervals; APQ5 needs >=6 (five interior amplitude points).
minimum=[2,3,3,6];mixed=nan(size(cycle));source=repmat("InvalidCycles_NoFallback",size(cycle));
for j=1:4
    real=isfinite(cycle(:,j)) & cycle(:,j)>=0;
    mixed(real,j)=cycle(real,j);source(real,j)="Cycle";
    missing=~real;tooFew=counts<minimum(j);
    source(missing & ~support,j)="NoPeriodicSupport";
    eligible=missing & tooFew & support & ~invalid;
    source(eligible,j)="InsufficientCycles_NoFallback";
    if ismember(window,cfg.ProxyFallbackWindowsMs)
        source(eligible & ~voiced,j)="Unvoiced_NoFallback";
        good=eligible & voiced & isfinite(proxy(:,j)) & proxy(:,j)>=0;
        source(eligible & voiced & ~isfinite(proxy(:,j)),j)="ProxyUnavailable";
        mixed(good,j)=proxy(good,j);source(good,j)="Proxy_InsufficientCycles";
    end
end
end

function t=vfDiagnosticTable(extra,proxy)
t=array2table(extra.CycleValues,'VariableNames',{'JitterLocal_CycleValue','JitterRAP_CycleValue','ShimmerLocal_CycleValue','ShimmerAPQ5_CycleValue'});
t=[t,array2table(proxy,'VariableNames',{'JitterLocal_ProxyValue','JitterRAP_ProxyValue','ShimmerLocal_ProxyValue','ShimmerAPQ5_ProxyValue'})];
t=[t,array2table(extra.Sources,'VariableNames',{'JitterLocal_Source','JitterRAP_Source','ShimmerLocal_Source','ShimmerAPQ5_Source'})];
t.CompleteCycles=extra.CompleteCycles;t.LongestValidCycleRun=extra.LongestValidCycleRun;t.CycleF0=extra.CycleF0;
end

function vfWriteText(file,text)
f=fopen(file,'w','n','UTF-8');assert(f>=0);clean=onCleanup(@()fclose(f));fprintf(f,'%s',text); %#ok<NASGU>
end

function vfRemoveOwnTemp(folder)
if isfolder(folder),try,rmdir(folder,'s');catch,end,end
end

function vfSelfTest(cfg)
extractorSelfTest();vfInputSelfTest();
% Independent analytic checks of the cycle formulas.
periods=[.005;.0051;.0049;.005;.00505;.00495];amps=[1;1.1;.9;1.05;1];
m=vfAcousticMetrics(periods,amps,cfg);
assert(abs(m(1)-100*mean(abs(diff(periods)))/mean(periods))<1e-10);
assert(abs(m(2)-100*mean(abs(periods(2:end-1)-(periods(1:end-2)+periods(2:end-1)+periods(3:end))/3))/mean(periods))<1e-10);
assert(abs(m(3)-100*mean(abs(diff(amps)))/mean(amps))<1e-10);
assert(abs(m(4)-100*abs(amps(3)-mean(amps))/mean(amps))<1e-10);
assert(all(isnan(vfAcousticMetrics([.005;.010],amps,cfg))));
[emptyPulses,~,emptyF0]=vfAcousticPulses(zeros(16000,1),16000,cfg);
assert(isempty(emptyPulses)&&all(isnan(emptyF0)),'Silence generated acoustic pulses.');

% A failed cycle estimate is not a license to replace it with a Proxy.
c=nan(4,4);p=repmat([.1,.2,.3,.4],4,1);n=[2;9;2;2];support=true(4,1);bad=[false;false;true;false];voiced=[true;true;true;false];
[m,s]=vfCombineCycleProxy(c,p,n,support,bad,voiced,25,cfg);
assert(isnan(m(1,1))&&all(m(1,2:4)==p(1,2:4)));
assert(all(isnan(m(2,:)))&&all(s(2,:)=="InvalidCycles_NoFallback"));
assert(all(isnan(m(3,:)))&&all(isnan(m(4,:))));
[m,~]=vfCombineCycleProxy(c,p,n,support,bad,voiced,75,cfg);assert(all(isnan(m),'all'));
c=repmat([0,.1,.2,.3],4,1);[m,s]=vfCombineCycleProxy(c,p,n,support,bad,voiced,25,cfg);
assert(isequal(m,c)&&all(s=="Cycle",'all'));
base=tempname;mkdir(base);cleanupHandle=onCleanup(@()vfRemoveOwnTemp(base)); %#ok<NASGU>
work=fullfile(base,['voice check ' char(214) ' space']);mkdir(work);
cfg.OutputFolder=fullfile(work,'features');mkdir(cfg.OutputFolder);
cfg.EngineVersion="MATLAB_YIN_AcousticCycles_v1";cfg.EngineSHA256=vfHashText(cfg.EngineVersion);
% Independent physical oracle: an exactly periodic clean tone has nearly
% zero cycle perturbation, while amplitude modulation must increase shimmer.
fs=50000;t=(0:fs-1)'/fs;clean=.35*sin(2*pi*320*t);am=(1+.15*sin(2*pi*17*t)).*clean;
audio=fullfile(work,'clean.wav');audiowrite(audio,clean,fs,'BitsPerSample',16);
ref=vfMatlabCycleRecord(audio,16000,cfg);before=dir(fullfile(cfg.OutputFolder,'_cycle_cache','*.mat'));
ag= vfMatlabCycleRecord(audio,16000,cfg);after=dir(fullfile(cfg.OutputFolder,'_cycle_cache','*.mat'));
assert(isequaln(ref,ag)&&numel(before)==numel(after),'Acoustic-cycle cache was not reused.');
r=ref.Metrics(ref.Metrics.Window_ms==75 & ref.Metrics.FrameID>10,:);
assert(median(r.JitterLocal,'omitnan')<.02&&median(r.ShimmerLocal,'omitnan')<.02,'Clean-tone cycle oracle failed.');
modulated=fullfile(work,'modulated.wav');audiowrite(modulated,am,fs,'BitsPerSample',16);
r2=vfMatlabCycleRecord(modulated,16000,cfg);q=r2.Metrics(r2.Metrics.Window_ms==75,:);
assert(median(q.ShimmerLocal,'omitnan')>.1,'Amplitude modulation was not detected.');
% Full extraction from native PCM/WAV metadata, with two disease memberships.
nativeNames=["Healthy_1-a_n","Hyperfunktionelle_106-a_n",("Reinke "+string(char(214))+"dem_106-a_n")];
labels=[0;1;1];diseases=["Healthy";"Hyperfunktionelle";("Reinke "+string(char(214))+"dem")];ids=["1";"106";"106"];
paths=strings(3,1);sources=strings(3,1);
for j=1:3
    paths(j)=fullfile(work,nativeNames(j)+".wav");
    frequency=120;if j>1,frequency=320;end
    signal=.35*sin(2*pi*frequency*t);pcm=int16(max(-32768,min(32767,round(signal*32768))));
    sourceFolder=fullfile(work,"source_"+j);mkdir(sourceFolder);sources(j)=fullfile(sourceFolder,ids(j)+"-a_n.nsp");
    vfWriteSyntheticNSP(sources(j),pcm,fs);audiowrite(paths(j),double(pcm)/32768,fs,'BitsPerSample',16);
end
T=table(paths,labels,["w";"m";"m"],diseases,ids,repmat("a",3,1),sources,repmat(fs,3,1),repmat(fs,3,1), ...
    'VariableNames',{'FilePath','Label','Gender','Disease','SubjectID','Vowel','SourceNSP','NativeSampleRate','OriginalSampleCount'});
metadata=fullfile(work,'Thesis_Final_Dataset.xlsx');writetable(T,metadata);
options=struct('DatasetPath',metadata,'OutputFolder',fullfile(work,'full_extract'), ...
    'Diseases',diseases','SaveMAT',true,'SaveCycleOnly',true,'Resume',true,'AutoSelfTest',false);
VoiceFeatureExtraction_NoPraat("run",options);
file=fullfile(options.OutputFolder,'FullFeats_25ms_Healthy.xlsx');A=readtable(file,'TextType','string','VariableNamingRule','preserve');
assert(height(A)==98 && width(A)==70,'Unexpected frame/schema dimensions.');
assert(any(A.ShimmerAPQ5_Source=="Proxy_InsufficientCycles"),'25ms insufficient-cycle fallback was not exercised.');
file75=fullfile(options.OutputFolder,'FullFeats_75ms_Healthy.xlsx');B=readtable(file75,'TextType','string','VariableNamingRule','preserve');
assert(height(B)==93 && ~any(startsWith(B.ShimmerAPQ5_Source,"Proxy_")));
P=readtable(fullfile(options.OutputFolder,'CycleOnly','FullFeats_25ms_Healthy.xlsx'),'TextType','string','VariableNamingRule','preserve');
proxyRows=A.ShimmerAPQ5_Source=="Proxy_InsufficientCycles";assert(all(isnan(P.ShimmerAPQ5(proxyRows))));
assert(all(isfinite(A.ShimmerAPQ5_CycleOrProxy(proxyRows))));
Q=readtable(fullfile(options.OutputFolder,'Feature_Extraction_QC.xlsx'));assert(height(Q)==6 && sum(Q.ProcessingErrors)==0);
F=readtable(fullfile(options.OutputFolder,'File_Extraction_QC.xlsx'),'TextType','string');assert(height(F)==6&&all(F.Status=="Completed"));
stamp=dir(file);VoiceFeatureExtraction_NoPraat("run",options);check=dir(file);assert(stamp.datenum==check.datenum,'Resume recomputed a completed group.');
% Source mutation must not silently reuse prior outputs.
[signal,rate]=audioread(paths(1));signal(20)=signal(20)+1/32768;audiowrite(paths(1),signal,rate,'BitsPerSample',16);
rejected=false;try,VoiceFeatureExtraction_NoPraat("run",options);catch,rejected=true;end
assert(rejected,'Mutated source content was silently reused.');
destination=getenv('VFE_TEST_OUTPUT');if ~isempty(destination),if ~isfolder(destination),mkdir(destination);end;copyfile(fullfile(options.OutputFolder,'*'),destination);end
fprintf('CYCLE_REFERENCE_FALLBACK_SCHEMA_MEMBERSHIP_RESUME_TESTS_PASSED\n');
end

function [x,fs]=vfReadNSP(file)
f=fopen(file,'rb','ieee-le');assert(f>=0,'Native NSP source not found: %s',file);c=onCleanup(@()fclose(f)); %#ok<NASGU>
fseek(f,0,'eof');bytes=ftell(f);fseek(f,0,'bof');magic=char(fread(f,8,'*uint8')');
assert(strcmp(magic,'FORMDS16'),'Expected FORMDS16 native source.');
size=fread(f,1,'uint32=>double');assert(isscalar(size)&&size==bytes-12,'Invalid NSP length.');
x=[];fs=[];expected=[];data=false;header=false;
while ftell(f)<bytes
    assert(bytes-ftell(f)>=8,'Truncated NSP chunk.');tag=char(fread(f,4,'*uint8')');n=fread(f,1,'uint32=>double');start=ftell(f);
    assert(isscalar(n)&&start+n<=bytes,'Truncated NSP payload.');
    if strcmp(tag,'HEDR')
        assert(~header&&n>=28);header=true;fseek(f,start+20,'bof');fs=fread(f,1,'uint32=>double');expected=fread(f,1,'uint32=>double');
    elseif strcmp(tag,'SDA_')
        assert(~data&&mod(n,2)==0);data=true;x=fread(f,n/2,'int16=>int16');
    elseif any(strcmp(tag,{'SDB_','SDAB'})),error('Multichannel NSP unsupported.');end
    fseek(f,start+n,'bof');
end
assert(header&&data&&numel(x)==expected,'NSP header/payload mismatch.');
end

function vfWriteSyntheticNSP(file,x,fs)
f=fopen(file,'wb','ieee-le');assert(f>=0);c=onCleanup(@()fclose(f)); %#ok<NASGU>
fwrite(f,'FORMDS16','char');fwrite(f,48+2*numel(x),'uint32');fwrite(f,'HEDR','char');fwrite(f,32,'uint32');
fwrite(f,'Jan 01 00:00:00 2000','char');fwrite(f,fs,'uint32');fwrite(f,numel(x),'uint32');fwrite(f,[32767,-1],'int16');
fwrite(f,'SDA_','char');fwrite(f,2*numel(x),'uint32');fwrite(f,x,'int16');
end


function vfSaveFeatures(file,Tout,settings)
tmp=[tempname(fileparts(file)) '.mat'];save(tmp,'Tout','settings','-v7.3');movefile(tmp,file,'f');
end

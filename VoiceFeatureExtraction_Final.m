function VoiceFeatureExtraction_Final(mode,overrides)
% Standalone MATLAB R2023b extractor for corrected native-PCM WAV inputs.
% Run: VoiceFeatureExtraction_Final
% All local functions are included below. Requires Signal Processing Toolbox.
% Cycle metrics: official Praat; Windows x64 dependency is downloaded/verified once.
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
[cfg.PraatPath,cfg.PraatVersion,cfg.PraatBinarySHA256]=vfEnsurePraat(cfg);
assert(cfg.PerturbationMode=="cycle_with_proxy_fallback" && isequal(cfg.ProxyFallbackWindowsMs,25),'This version supports cycle metrics with 25ms-only insufficient-cycle fallback.');
assert(isequal(cfg.WindowMs,[25,75]) && cfg.TargetFs==16000,'This validated version uses 25/75ms at 16kHz.');
sourceHash=vfHashFile(string(mfilename('fullpath'))+".m");
settings=cfg;settings.CodeSHA256=sourceHash;settings.FeatureNames=vfFeatureNames(cfg);settings.InputMetadataSHA256=vfHashFile(cfg.DatasetPath);
settings.ProxyEffectiveSupportMs=cfg.WindowMs+2*cfg.PerturbRadius*cfg.HopMs;
settings.CycleMetricPercentUnits=true;settings.PraatCycleContext="Pulses tracked from native-rate whole recording; queries restricted to each frame interval";
settings.MinimumPointProcessIntervals=[2,3,3,6];
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
                proxy=F(:,7:10);reference=vfPraatRecord(path,numel(y),cfg);
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
fprintf('Finished. Inspect Feature_Extraction_QC.xlsx and File_Extraction_QC.xlsx before modeling.\n');
end

function cfg=vfConfiguration
cfg=struct('DatasetPath','D:\SVD\meta\svd_wav\data\Thesis_Final_Dataset.xlsx', ...
    'OutputFolder','D:\SVD\meta\svd_wav\Features_Final_CycleFallback', ...
    'Diseases',["Healthy","Rekurrensparese","Hyperfunktionelle","Laryngitis","Reinke Ödem","Spasmodische Dysphonie","Stimmlippenpolyp"], ...
    'TargetFs',16000,'WindowMs',[25,75],'HopMs',10,'NFFT',2048,'PerturbRadius',2, ...
    'PerturbationMode',"cycle_with_proxy_fallback",'ProxyFallbackWindowsMs',25, ...
    'PraatPath',"",'AutoDownloadPraat',true,'PraatVersion',"",'PraatBinarySHA256',"", ...
    'PitchFloor',50,'PitchCeiling',500,'MaxPeriodFactor',1.3,'MaxAmplitudeFactor',1.6, ...
    'VerifyNSPPayload',true,'SaveCycleOnly',true,'Resume',true,'SaveMAT',true,'Version',"Native_PCM_CycleOrProxy43_v4");
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
for f=1:nFrames, if voicedMask(f), hnr(f)=estimateHNRAutocorr(frames(:,f).*windowVector,fs,f0(f)); end, end

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

function hnrDb=estimateHNRAutocorr(x,fs,f0)
hnrDb=NaN; if ~isfinite(f0)||f0<=0, return; end
x=double(x(:)); x=x-mean(x); if sum(x.^2)<eps, return; end
r=xcorr(x,'coeff'); mid=length(x); pr=r(mid:end);
lag0=round(fs/f0); sr=max(1,round(0.15*lag0));
loLag=max(1,lag0-sr); hiLag=min(numel(pr)-1,lag0+sr);
if loLag>hiLag, return; end
% pr(1) is lag zero: physical lag L is stored at MATLAB index L+1.
rPeak=max(pr(loLag+1:hiLag+1));
rPeak=min(max(rPeak,eps),1-eps); hnrDb=10*log10(rPeak/(1-rPeak));
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
% Peak exactly at the upper search boundary detects the HNR lag/index shift.
x=zeros(400,1); x(150)=1; x(242)=1;
x=x-mean(x); [corr,lags]=xcorr(x,'coeff');
peak=max(corr(lags>=68 & lags<=92)); peak=min(max(peak,eps),1-eps);
referenceHNR=10*log10(peak/(1-peak));
actualHNR=estimateHNRAutocorr(x,fs,200);
assert(abs(actualHNR-referenceHNR)<1e-10,'HNR lag-boundary regression.');
assert(isnan(estimateHNRAutocorr(zeros(400,1),fs,200)));
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
settings.Resume=false;blob=jsonencode(settings)+newline;
for j=1:height(T)
    p=T.FilePath(j);d=dir(p);
    if isempty(d),fileInfo="MISSING";else,fileInfo=vfHashFile(p);end
    blob=blob+strjoin([p,string(T.Label(j)),T.Gender(j),T.Disease(j),T.SubjectID(j),T.Vowel(j), ...
        string(T.NativeSampleRate(j)),string(T.OriginalSampleCount(j)),fileInfo],"|")+newline;
end
md=java.security.MessageDigest.getInstance('SHA-256');md.update(unicode2native(char(blob),'UTF-8'));signature=vfDigest(md);
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
cfg=vfConfiguration();T=table(["/tmp/Healthy_1-a_n.wav";"/tmp/Rekurrensparese_106-a_n.wav";"/tmp/Reinke Ödem_106-a_n.wav"], ...
    [0;1;1],["w";"m";"m"],["Healthy";"Rekurrensparese";"Reinke Ödem"],["1";"106";"106"],["a";"a";"a"], ...
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

function [exe,version,hash]=vfEnsurePraat(cfg)
exe=string(cfg.PraatPath);
if strlength(exe)==0,exe=string(getenv('PRAAT_BIN'));end
if strlength(exe)==0
    candidate=fullfile(fileparts(mfilename('fullpath')),'Praat.exe');
    if isfile(candidate),exe=string(candidate);end
end
if strlength(exe)==0
    install=fullfile(cfg.OutputFolder,'_praat_runtime');exe=string(fullfile(install,'Praat.exe'));
    if ~isfile(exe)
        assert(ispc&&contains(computer('arch'),'64')&&cfg.AutoDownloadPraat, ...
            'Set PraatPath to an installed Praat executable; Windows x64 supports automatic download.');
        if ~isfolder(install),mkdir(install);end
        url='https://github.com/praat/praat.github.io/releases/download/v7.0.03/praat7003_win-x64v1.zip';
        expected="a8795db140d7544bbf43ee9e9f538311c0de33ad4135f69a3d8560fb78504fae";
        archive=fullfile(install,'praat7003_win-x64v1.zip');
        fprintf('Downloading official Praat 7.0.03 once (about 52 MB)...\n');
        try
            websave(archive,url,weboptions('Timeout',180));
        catch err
            error('Praat download failed. Download the official Windows x64v1 ZIP and place Praat.exe beside this MATLAB file. Details: %s',err.message);
        end
        assert(vfHashFile(archive)==expected,'Praat archive checksum mismatch; archive was not executed.');
        unzip(archive,install);assert(isfile(exe),'Praat.exe not found after extraction.');
    end
end
assert(isfile(exe),'Praat executable not found: %s',exe);
[status,text]=system(vfCommand([exe,"--version"]));assert(status==0&&contains(text,'Praat'),'Cannot execute Praat: %s',text);
version=strip(string(text));token=regexp(char(version),'Praat\s+(\d+)\.','tokens','once');
assert(~isempty(token)&&str2double(token{1})>=6,'Unsupported Praat version.');
hash=vfHashFile(exe);
end

function reference=vfPraatRecord(wav,nResampled,cfg)
cache=fullfile(cfg.OutputFolder,'_cycle_cache');if ~isfolder(cache),mkdir(cache);end
script=vfPraatScript();audioHash=vfHashFile(wav);
counts=max(0,floor((nResampled-round(cfg.WindowMs*cfg.TargetFs/1000))/round(cfg.HopMs*cfg.TargetFs/1000))+1);
key=vfHashText(audioHash+cfg.PraatBinarySHA256+vfHashText(script)+jsonencode( ...
    {cfg.TargetFs,cfg.WindowMs,cfg.HopMs,counts,cfg.PitchFloor,cfg.PitchCeiling,cfg.MaxPeriodFactor,cfg.MaxAmplitudeFactor}));
file=fullfile(cache,key+".mat");
if isfile(file)
    z=load(file,'reference');reference=z.reference;
    assert(reference.Signature==key&&isequal(reference.FrameCounts,counts),'Clinical cache mismatch.');return;
end
work=tempname(cache);mkdir(work);cleanup=onCleanup(@()vfRemoveOwnTemp(work)); %#ok<NASGU>
input=fullfile(work,'input.wav');scriptFile=fullfile(work,'cycle_metrics.praat');
output=fullfile(work,'metrics.csv');pulse=fullfile(work,'pulses.txt');
copyfile(wav,input);vfWriteText(scriptFile,script);
token=regexp(char(cfg.PraatVersion),'Praat\s+(\d+)\.','tokens','once');flags=strings(1,0);
if str2double(token{1})>=7,flags="--FULL-TRUST";end % own generated script only: audio read and local result writes
args=[string(cfg.PraatPath),flags,"--run",string(scriptFile),string(input),string(output),string(pulse), ...
    string(counts(1)),string(counts(2)),string(cfg.HopMs/1000),string(cfg.PitchFloor), ...
    string(cfg.PitchCeiling),string(cfg.MaxPeriodFactor),string(cfg.MaxAmplitudeFactor)];
[status,message]=system(vfCommand(args));
assert(status==0&&isfile(output)&&isfile(pulse),'Praat cycle analysis failed; no Proxy fallback for runtime failure. %s',message);
R=readtable(output,'TextType','string','TreatAsMissing',{'--undefined--','undefined'});
expected=["Window_ms","FrameID","JitterLocal","JitterRAP","ShimmerLocal","ShimmerAPQ5","PraatF0"];
assert(all(ismember(expected,string(R.Properties.VariableNames))),'Malformed Praat output columns.');
for column=expected,R.(column)=str2double(string(R.(column)));end
assert(height(R)==sum(counts),'Praat frame count mismatch.');
for j=1:2
    r=R(R.Window_ms==cfg.WindowMs(j),:);
    assert(height(r)==counts(j)&&isequal(r.FrameID,(1:counts(j))'),'Praat frame identifiers mismatch.');
end
lines=splitlines(string(fileread(pulse)));numbers=str2double(strip(lines));numbers=numbers(isfinite(numbers));
assert(numel(numbers)>=3&&numbers(3)==round(numbers(3))&&numel(numbers)==numbers(3)+3,'Malformed PointProcess output.');
points=numbers(4:end);assert(all(diff(points)>0),'Non-increasing pulse times.');
reference=struct('Signature',key,'FrameCounts',counts,'Metrics',R,'Pulses',points, ...
    'AudioSHA256',audioHash,'PraatVersion',cfg.PraatVersion,'PraatSHA256',cfg.PraatBinarySHA256);
vfAtomicSave(file,'reference',reference);
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
    'LongestValidCycleRun',validRuns,'PraatF0',R.PraatF0);
end

function [mixed,source]=vfCombineCycleProxy(cycle,proxy,counts,support,invalid,voiced,window,cfg)
% Praat amplitude estimates need bordering pulses: shimmer local requires
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
t.CompleteCycles=extra.CompleteCycles;t.LongestValidCycleRun=extra.LongestValidCycleRun;t.PraatF0=extra.PraatF0;
end

function command=vfCommand(args)
args=string(args);quoted=cell(1,numel(args));sq=char(39);dq=char(34);
for j=1:numel(args)
    a=char(args(j));assert(~any(a==char(10)|a==char(13)),'Newline in command argument.');
    if ispc
        assert(~contains(a,dq)&&~contains(a,'%'),'Use paths without double quotes or percent signs.');
        quoted{j}=[dq,a,dq];
    else
        quoted{j}=[sq,strrep(a,sq,[sq,dq,sq,dq,sq]),sq];
    end
end
command=strjoin(quoted,' ');
if ispc,command=['cmd /D /V:OFF /S /C "' command '"'];end
end

function vfWriteText(file,text)
f=fopen(file,'w','n','UTF-8');assert(f>=0);clean=onCleanup(@()fclose(f));fprintf(f,'%s',text); %#ok<NASGU>
end

function vfRemoveOwnTemp(folder)
if isfolder(folder),try,rmdir(folder,'s');catch,end,end
end

function script=vfPraatScript
lines=[ ...
    "form Frame cycle measurements"; ...
    "    sentence input_file"; ...
    "    sentence output_file"; ...
    "    sentence pulse_file"; ...
    "    integer frames25 1"; ...
    "    integer frames75 1"; ...
    "    positive hop_seconds 0.01"; ...
    "    positive pitch_floor 50"; ...
    "    positive pitch_ceiling 500"; ...
    "    positive max_period_factor 1.3"; ...
    "    positive max_amplitude_factor 1.6"; ...
    "endform"; ...
    "Read from file: input_file$"; ...
    "snd = selected(""Sound"")"; ...
    "To PointProcess (periodic, cc): pitch_floor, pitch_ceiling"; ...
    "pp = selected(""PointProcess"")"; ...
    "Save as short text file: pulse_file$"; ...
    "selectObject: snd"; ...
    "To Pitch (ac): hop_seconds, pitch_floor, 15, ""yes"", 0.03, 0.45, 0.01, 0.35, 0.14, pitch_ceiling"; ...
    "pit = selected(""Pitch"")"; ...
    "writeFileLine: output_file$, ""Window_ms,FrameID,JitterLocal,JitterRAP,ShimmerLocal,ShimmerAPQ5,PraatF0"""; ...
    "for w from 1 to 2"; ...
    "    if w = 1"; ...
    "        win = 25"; ...
    "        n = frames25"; ...
    "    else"; ...
    "        win = 75"; ...
    "        n = frames75"; ...
    "    endif"; ...
    "    for frame from 1 to n"; ...
    "        t0 = (frame - 1) * hop_seconds"; ...
    "        t1 = t0 + win / 1000"; ...
    "        selectObject: pp"; ...
    "        jl = Get jitter (local): t0, t1, 1 / pitch_ceiling, 1 / pitch_floor, max_period_factor"; ...
    "        jr = Get jitter (rap): t0, t1, 1 / pitch_ceiling, 1 / pitch_floor, max_period_factor"; ...
    "        selectObject: snd, pp"; ...
    "        sl = Get shimmer (local): t0, t1, 1 / pitch_ceiling, 1 / pitch_floor, max_period_factor, max_amplitude_factor"; ...
    "        sa = Get shimmer (apq5): t0, t1, 1 / pitch_ceiling, 1 / pitch_floor, max_period_factor, max_amplitude_factor"; ...
    "        selectObject: pit"; ...
    "        f0 = Get value at time: (t0 + t1) / 2, ""Hertz"", ""Linear"""; ...
    "        appendFileLine: output_file$, win, "","", frame, "","", fixed$(100 * jl, 15), "","", fixed$(100 * jr, 15), "","", fixed$(100 * sl, 15), "","", fixed$(100 * sa, 15), "","", fixed$(f0, 15)"; ...
    "    endfor"; ...
    "endfor"; ...
];
script=strjoin(lines,newline)+newline;
end

function vfSelfTest(cfg)
extractorSelfTest();vfInputSelfTest();
% A failed cycle estimate is not a license to replace it with a Proxy.
c=nan(4,4);p=repmat([.1,.2,.3,.4],4,1);n=[2;9;2;2];support=true(4,1);bad=[false;false;true;false];voiced=[true;true;true;false];
[m,s]=vfCombineCycleProxy(c,p,n,support,bad,voiced,25,cfg);
assert(isnan(m(1,1))&&all(m(1,2:4)==p(1,2:4)));
assert(all(isnan(m(2,:)))&&all(s(2,:)=="InvalidCycles_NoFallback"));
assert(all(isnan(m(3,:)))&&all(isnan(m(4,:))));
[m,~]=vfCombineCycleProxy(c,p,n,support,bad,voiced,75,cfg);assert(all(isnan(m),'all'));
c=repmat([0,.1,.2,.3],4,1);[m,s]=vfCombineCycleProxy(c,p,n,support,bad,voiced,25,cfg);
assert(isequal(m,c)&&all(s=="Cycle",'all'));
base=tempname;mkdir(base);work=fullfile(base,'voice check Ö space');mkdir(work);
cfg.OutputFolder=fullfile(work,'features');mkdir(cfg.OutputFolder);
[cfg.PraatPath,cfg.PraatVersion,cfg.PraatBinarySHA256]=vfEnsurePraat(cfg);
% Independent physical oracle: an exactly periodic clean tone has nearly
% zero cycle perturbation, while amplitude modulation must increase shimmer.
fs=50000;t=(0:fs-1)'/fs;clean=.35*sin(2*pi*320*t);am=(1+.15*sin(2*pi*17*t)).*clean;
audio=fullfile(work,'clean.wav');audiowrite(audio,clean,fs,'BitsPerSample',16);
ref=vfPraatRecord(audio,16000,cfg);before=dir(fullfile(cfg.OutputFolder,'_cycle_cache','*.mat'));
ag= vfPraatRecord(audio,16000,cfg);after=dir(fullfile(cfg.OutputFolder,'_cycle_cache','*.mat'));
assert(isequaln(ref,ag)&&numel(before)==numel(after),'Clinical cache was not reused.');
r=ref.Metrics(ref.Metrics.Window_ms==75 & ref.Metrics.FrameID>10,:);
assert(median(r.JitterLocal,'omitnan')<.02&&median(r.ShimmerLocal,'omitnan')<.02,'Clean-tone cycle oracle failed.');
modulated=fullfile(work,'modulated.wav');audiowrite(modulated,am,fs,'BitsPerSample',16);
r2=vfPraatRecord(modulated,16000,cfg);q=r2.Metrics(r2.Metrics.Window_ms==75,:);
assert(median(q.ShimmerLocal,'omitnan')>.1,'Amplitude modulation was not detected.');
% Full extraction from native PCM/WAV metadata, with two disease memberships.
nativeNames=["Healthy_1-a_n","Hyperfunktionelle_106-a_n","Reinke Ödem_106-a_n"];
labels=[0;1;1];diseases=["Healthy";"Hyperfunktionelle";"Reinke Ödem"];ids=["1";"106";"106"];
paths=strings(3,1);sources=strings(3,1);
for j=1:3
    paths(j)=fullfile(work,nativeNames(j)+".wav");
    frequency=120;if j>1,frequency=320;end
    signal=.35*sin(2*pi*frequency*t);pcm=int16(max(-32768,min(32767,round(signal*32768))));
    vfWriteSyntheticNSP(fullfile(work,ids(j)+"_source_"+j+".nsp"),pcm,fs); % reference metadata below uses a parseable basename
    sourceFolder=fullfile(work,"source_"+j);mkdir(sourceFolder);sources(j)=fullfile(sourceFolder,ids(j)+"-a_n.nsp");
    vfWriteSyntheticNSP(sources(j),pcm,fs);audiowrite(paths(j),double(pcm)/32768,fs,'BitsPerSample',16);
end
T=table(paths,labels,["w";"m";"m"],diseases,ids,repmat("a",3,1),sources,repmat(fs,3,1),repmat(fs,3,1), ...
    'VariableNames',{'FilePath','Label','Gender','Disease','SubjectID','Vowel','SourceNSP','NativeSampleRate','OriginalSampleCount'});
metadata=fullfile(work,'Thesis_Final_Dataset.xlsx');writetable(T,metadata);
options=struct('DatasetPath',metadata,'OutputFolder',fullfile(work,'full_extract'), ...
    'Diseases',diseases','PraatPath',cfg.PraatPath,'SaveMAT',true,'SaveCycleOnly',true,'Resume',true);
VoiceFeatureExtraction_Final("run",options);
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
stamp=dir(file);VoiceFeatureExtraction_Final("run",options);check=dir(file);assert(stamp.datenum==check.datenum,'Resume recomputed a completed group.');
% Source mutation must not silently reuse prior outputs.
[signal,rate]=audioread(paths(1));signal(20)=signal(20)+1/32768;audiowrite(paths(1),signal,rate,'BitsPerSample',16);
rejected=false;try,VoiceFeatureExtraction_Final("run",options);catch,rejected=true;end
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

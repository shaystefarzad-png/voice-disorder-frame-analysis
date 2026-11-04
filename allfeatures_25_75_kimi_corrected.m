function allfeatures_25_75_kimi_corrected(mode)
% Run extraction: allfeatures_25_75_kimi_corrected
% Numerical regression checks: allfeatures_25_75_kimi_corrected("selftest")
% MATLAB R2023b + Signal Processing Toolbox. No actual WAVs are bundled.
if nargin==0, mode="run"; end
if strcmpi(string(mode),"selftest")
    extractorSelfTest();
    return
end
assert(strcmpi(string(mode),"run"),'Mode must be run or selftest.');
clc;
%% ========================================================================
% CORRECTED FEATURE EXTRACTOR - 43 frame-level acoustic features
% 25 ms and 75 ms windows, all diseases, subject metadata, hybrid-ready.
%% ========================================================================

%% USER SETTINGS
datasetPath = 'D:\SVD\meta\svd_wav\data\Thesis_Final_Dataset.xlsx';
outputFolder = 'D:\SVD\meta\svd_wav\Features_Corrected';
% Never mix the regenerated workbooks with the original feature version.
extractorVersion = 'YIN_HNR_LPC_corrected_v2_2026_11_04';
showProgress = usejava('desktop');
allowedDiseases = ["Healthy","Rekurrensparese","Hyperfunktionelle", ...
    "Laryngitis","Reinke Ödem","Spasmodische Dysphonie","Stimmlippenpolyp"];
targetFs = 16000;
windowMs = [25 75];
% IMPORTANT: keep hop identical so window length is the only changed factor
% and rows can be paired by FileName + FrameStartTime.
hopMs = [10 10];
assert(numel(unique(hopMs))==1,'Hop must be identical for all window lengths.');
f0Min = 50; f0Max = 500;
yinThreshold = 0.20;
minVoicingConfidence = 0.70;
vadRelativeThreshold = 0.01;
nfft = 2048;
rolloffPercent = 0.85;
numMFCC = 13;
numMelFilters = 26;
perturbRadius = 2;
saveQC = true;
saveExcel = true;  % human-readable copy
saveMat = true;    % fast/full-precision primary output (recommended)

if ~exist(outputFolder,'dir'), mkdir(outputFolder); end
rng(42,'twister');

%% FEATURE NAMES
featureNames = {'ZCR','ZCR_Var','RMS','PeakAmp','Entropy','EnergyEntropy', ...
    'JitterProxy_local','JitterProxy_rap','ShimmerProxy_local','ShimmerProxy_apq', ...
    'HNR','SpecCentroid','SpecRolloff','SpecFlux','SpecEntropy', ...
    'MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13', ...
    'DeltaMFCC1','DeltaMFCC2','DeltaMFCC3','DeltaMFCC4','DeltaMFCC5','DeltaMFCC6','DeltaMFCC7','DeltaMFCC8','DeltaMFCC9','DeltaMFCC10','DeltaMFCC11','DeltaMFCC12','DeltaMFCC13', ...
    'FormantF1','FormantF2'};
numFeatures = numel(featureNames);
assert(numFeatures==43,'Expected exactly 43 features.');
metaNames = {'FileName','SubjectID','Vowel','Label','Gender','Disease', ...
    'FrameID','FrameStartTime','FrameCenterTime','IsActive','IsVoiced','VoicingConfidence'};

settings = struct('TargetFs',targetFs,'WindowMs',windowMs,'HopMs',hopMs, ...
    'F0Min',f0Min,'F0Max',f0Max,'YINThreshold',yinThreshold, ...
    'MinVoicingConfidence',minVoicingConfidence,'VADRelativeThreshold',vadRelativeThreshold, ...
    'NFFT',nfft,'RolloffPercent',rolloffPercent,'NumMFCC',numMFCC, ...
    'NumMelFilters',numMelFilters,'PerturbRadius',perturbRadius, ...
    'FeatureNames',{featureNames},'ExtractorVersion',extractorVersion, ...
    'TemporalContextHopRadius',2);

%% READ DATASET
if ~exist(datasetPath,'file'), error('Dataset not found: %s',datasetPath); end
T = readtable(datasetPath,'VariableNamingRule','preserve');
requiredVars = {'FilePath','Label','Gender','Disease'};
for k=1:numel(requiredVars)
    if ~ismember(requiredVars{k},T.Properties.VariableNames)
        error('Missing required column: %s',requiredVars{k});
    end
end
assert(isnumeric(T.Label) || islogical(T.Label),'Label must be numeric 0/1.');
T.FilePath = string(T.FilePath);
T.Gender = string(T.Gender);
T.Disease = string(T.Disease);
validRows = ~ismissing(T.FilePath) & ~ismissing(T.Disease) & ...
    strlength(strtrim(T.FilePath))>0 & strlength(strtrim(T.Disease))>0 & isfinite(double(T.Label));
T = T(validRows,:);
assert(all(ismember(double(T.Label),[0 1])),'Only binary Label=0/1 is supported.');
T.Disease = strtrim(T.Disease);
T = T(ismember(lower(T.Disease),lower(allowedDiseases)),:);
assert(~isempty(T),'No requested disease groups were found.');
healthy = strcmpi(T.Disease,'Healthy');
assert(all(double(T.Label(healthy))==0) && all(double(T.Label(~healthy))==1), ...
    'Healthy must have Label=0 and disease groups Label=1.');
[~,uIdx] = unique(lower(strtrim(T.FilePath)),'stable');
numDuplicates = height(T)-numel(uIdx);
T = T(uIdx,:);
fprintf('Rows after cleanup: %d\n',height(T));
fprintf('Duplicate FilePath rows removed: %d\n',numDuplicates);
diseases = unique(T.Disease,'stable');
fprintf('Disease groups: %d\n',numel(diseases)); disp(diseases);
QC = table();
save(fullfile(outputFolder,'Extraction_Settings.mat'),'settings');

%% MAIN LOOP
for w=1:numel(windowMs)
    win_ms = windowMs(w); hop_ms = hopMs(w);
    winLen = round(win_ms/1000*targetFs);
    hopLen = round(hop_ms/1000*targetFs);
    winVector = hamming(winLen,'periodic');
    melBank = createMelFilterBank(targetFs,nfft,numMelFilters,0,targetFs/2);

    fprintf('\n===== Window %d ms | Hop %d ms =====\n',win_ms,hop_ms);

    for d=1:numel(diseases)
        diseaseName = diseases(d);
        Td = T(strcmpi(T.Disease,diseaseName),:);
        % One block per recording avoids repeated reallocation of large arrays.
        metaBlocks = cell(height(Td),1);
        featBlocks = cell(height(Td),1);
        diseaseQC = initializeQC(win_ms,diseaseName);
        fprintf('[%d ms] %s : %d recordings\n',win_ms,diseaseName,height(Td));
        hWait = [];
        if showProgress, hWait = waitbar(0,sprintf('%d ms - %s',win_ms,diseaseName)); end

        for i=1:height(Td)
            try
                filePath = char(Td.FilePath(i));
                if ~exist(filePath,'file')
                    diseaseQC.MissingFiles = diseaseQC.MissingFiles + 1; continue;
                end
                label = double(Td.Label(i)); gender = char(Td.Gender(i)); disease = char(Td.Disease(i));
                [~,fileName,~] = fileparts(filePath);
                [subjectID,vowel] = parseSVDFileName(fileName);
                [y,fsRead] = audioread(filePath);
                if isempty(y), diseaseQC.EmptyFiles = diseaseQC.EmptyFiles + 1; continue; end
                if size(y,2)>1, y = mean(y,2); end
                y = double(y(:));
                if fsRead~=targetFs, y = resample(y,targetFs,fsRead); end
                y = y - mean(y,'omitnan');
                y(~isfinite(y)) = 0;
                [frames,starts] = frameSignalNoPadding(y,winLen,hopLen);
                if size(frames,2)<3
                    diseaseQC.TooShortFiles = diseaseQC.TooShortFiles + 1; continue;
                end
                frameStartTime = (starts-1)/targetFs;
                frameCenterTime = ((starts-1)+(winLen-1)/2)/targetFs;

                [featMat,activeMask,voicedMask,voiceConf,frameQC] = extract43Features(frames,targetFs,winVector,nfft,rolloffPercent,melBank,numMFCC,f0Min,f0Max,yinThreshold,minVoicingConfidence,vadRelativeThreshold,perturbRadius);
                nFrames = size(featMat,1);
                blockMeta = cell(nFrames,numel(metaNames));
                for fr=1:nFrames
                    blockMeta{fr,1}=fileName;
                    blockMeta{fr,2}=subjectID;
                    blockMeta{fr,3}=vowel;
                    blockMeta{fr,4}=label;
                    blockMeta{fr,5}=gender;
                    blockMeta{fr,6}=disease;
                    blockMeta{fr,7}=fr;
                    blockMeta{fr,8}=frameStartTime(fr);
                    blockMeta{fr,9}=frameCenterTime(fr);
                    blockMeta{fr,10}=logical(activeMask(fr));
                    blockMeta{fr,11}=logical(voicedMask(fr));
                    blockMeta{fr,12}=voiceConf(fr);
                end
                metaBlocks{i} = blockMeta;
                featBlocks{i} = featMat;
                diseaseQC.FilesProcessed = diseaseQC.FilesProcessed + 1;
                diseaseQC.TotalFrames = diseaseQC.TotalFrames + nFrames;
                diseaseQC.ActiveFrames = diseaseQC.ActiveFrames + sum(activeMask);
                diseaseQC.VoicedFrames = diseaseQC.VoicedFrames + sum(voicedMask);
                diseaseQC.NaNFeatureValues = diseaseQC.NaNFeatureValues + sum(isnan(featMat),'all');
                diseaseQC.InfFeatureValues = diseaseQC.InfFeatureValues + sum(isinf(featMat),'all');
                diseaseQC.PitchOutliersRejected = diseaseQC.PitchOutliersRejected + frameQC.PitchOutliersRejected;
            catch ME
                diseaseQC.ProcessingErrors = diseaseQC.ProcessingErrors + 1;
                warning('Error in %s: %s',char(Td.FilePath(i)),ME.message);
            end
            if ~isempty(hWait) && isgraphics(hWait), waitbar(i/height(Td),hWait); end
        end
        if ~isempty(hWait) && isgraphics(hWait), close(hWait); end

        nonEmpty = ~cellfun(@isempty,featBlocks);
        if ~any(nonEmpty)
            warning('No valid frames for %s at %d ms.',diseaseName,win_ms);
            QC = [QC; struct2table(diseaseQC)]; %#ok<AGROW>
            continue;
        end

        allMeta = vertcat(metaBlocks{nonEmpty});
        allFeat = vertcat(featBlocks{nonEmpty});

        Tmeta = cell2table(allMeta,'VariableNames',metaNames);
        Tfeat = array2table(allFeat,'VariableNames',featureNames);
        Tout = [Tmeta Tfeat];
        safeDiseaseName = regexprep(char(diseaseName),'[<>:"/\\|?*]','_');
        outputBase = fullfile(outputFolder,sprintf('FullFeats_%dms_%s',win_ms,safeDiseaseName));
        diseaseQC.ActivePercent = 100*diseaseQC.ActiveFrames/max(diseaseQC.TotalFrames,1);
        diseaseQC.VoicedPercent = 100*diseaseQC.VoicedFrames/max(diseaseQC.TotalFrames,1);
        diseaseQC.OutputRows = height(Tout);
        finiteFeatureCount = sum(isfinite(allFeat),1);
        diseaseQC.FeaturesWithNoFiniteData = sum(finiteFeatureCount==0);
        QC = [QC; struct2table(diseaseQC)]; %#ok<AGROW>

        if saveMat
            matFile = [outputBase '.mat'];
            save(matFile,'Tout','settings','diseaseQC','-v7.3');
        end
        if saveExcel
            outputFile = [outputBase '.xlsx'];
            writetable(Tout,outputFile);
        else
            outputFile = [outputBase '.mat'];
        end

        fprintf(' Saved: %s\n',outputFile);
        fprintf(' Frames=%d | Active=%.2f%% | Voiced=%.2f%% | NaN=%d | Inf=%d\n', ...
            height(Tout),diseaseQC.ActivePercent,diseaseQC.VoicedPercent, ...
            diseaseQC.NaNFeatureValues,diseaseQC.InfFeatureValues);
    end
end

if saveQC && ~isempty(QC)
    qcFile = fullfile(outputFolder,'Feature_Extraction_QC.xlsx');
    writetable(QC,qcFile); fprintf('\nQC saved: %s\n',qcFile);
end
fprintf('\nDONE. 43 features generated for 25 ms and 75 ms.\n');
fprintf('Do not normalize/impute using full data; fit those steps on TRAINING only.\n');

end

%% ======================= LOCAL FUNCTIONS ================================
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
    lo=max(1,f-perturbRadius); hi=min(nFrames,f+perturbRadius); idx=lo:hi; idx=idx(voicedMask(idx));
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

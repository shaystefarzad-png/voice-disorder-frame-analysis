function NSP_to_WAV_Corrected(mode,varargin)
% Native-rate, lossless PCM16 conversion. Does not treat NSP headers as audio.
% Dataset: NSP_to_WAV_Corrected
% One file: NSP_to_WAV_Corrected("one", "106-a_n.nsp", "106-a_n.wav")
% Checks: NSP_to_WAV_Corrected("selftest")
if nargin==0,mode="run";end
switch string(mode)
    case "selftest",nspSelfTest();return;
    case "one"
        assert(numel(varargin)==2,'Supply input NSP and output WAV paths.');
        [x,fs]=readNSP16(varargin{1});writeNativeWAV(varargin{2},x,fs);return;
    case "run"
    otherwise,error('Unknown conversion mode.');
end
% Settings: existing files and original research reports are never replaced.
csvFilePath='D:\SVD\meta\svd_wav\svd_labels_onlyN.csv';
rootSearchFolder='D:\SVD\meta\svd_wav';
outputWavFolder='D:\SVD\meta\svd_wav\data_NSP_Corrected';
allowed=["Healthy","Hyperfunktionelle","Laryngitis","Rekurrensparese", ...
    "Reinke Ödem","Spasmodische Dysphonie","Stimmlippenpolyp"];
assert(isfile(csvFilePath),'CSV not found: %s',csvFilePath);
if ~isfolder(outputWavFolder),mkdir(outputWavFolder);end
files=dir(fullfile(rootSearchFolder,'**','*.nsp'));assert(~isempty(files),'No NSP files found.');
index=containers.Map('KeyType','char','ValueType','any');
for j=1:numel(files)
    [~,name]=fileparts(files(j).name);key=nspNameKey(name);path=fullfile(files(j).folder,files(j).name);
    if isKey(index,key),q=index(key);q{end+1}=path;index(key)=q;else,index(key)={path};end
end
% CSV contract supplied by the project: first three fields = path,label,gender.
% readtable handles quoted commas; strsplit does not.
t=readtable(csvFilePath,'FileType','text','Delimiter',',','ReadVariableNames',false,'TextType','string');
assert(width(t)>=3,'CSV needs at least path, binary label, and gender fields.');
rows=cell(0,9);logs=cell(0,5);seen=containers.Map('KeyType','char','ValueType','char');
for i=1:height(t)
    source="";destination="";
    try
        raw=strip(string(t{i,1}));label=str2double(string(t{i,2}));gender=strip(string(t{i,3}));
        if i==1 && isnan(label) && any(strcmpi(strip(string(t{i,2})),["Label","Class","Target"]))
            logs(end+1,:)={i,"","","HeaderSkipped",""};continue; %#ok<AGROW>
        end
        assert(isscalar(label)&&ismember(label,[0,1]),'Label must be exactly 0 or 1.');
        source=string(resolveNSP(raw,index));
        d1=detectNSPDisease(raw);d2=detectNSPDisease(source);
        if strlength(d1)>0 && strlength(d2)>0,assert(d1==d2,'CSV path and actual source have different diseases.');end
        disease=d2;if strlength(disease)==0,disease=d1;end
        if label==0
            assert(strlength(disease)==0||disease=="Healthy",'Healthy label conflicts with disease folder.');disease="Healthy";
        else
            assert(strlength(disease)>0 && disease~="Healthy",'Patient disease is unknown or conflicts with Healthy folder.');
        end
        if ~ismember(disease,allowed)
            logs(end+1,:)={i,source,"","ExcludedDisease",disease};continue; %#ok<AGROW>
        end
        [~,name]=fileparts(source);tokens=regexp(char(name),'(?:^|_)([0-9]+)-([aiuAIU])_n$','tokens','once');
        assert(~isempty(tokens),'Expected subject-vowel_n sustained-vowel filename.');
        subject=string(tokens{1});vowel=lower(string(tokens{2}));
        destination=string(fullfile(outputWavFolder,char(disease+"_"+name+".wav")));
        key=lower(char(destination));
        if isKey(seen,key)
            assert(strcmpi(seen(key),char(source)),'Two different NSP sources target the same WAV.');
            error('Duplicate CSV recording: no extra metadata row was created.');
        end
        [x,fs]=readNSP16(source);writeNativeWAV(destination,x,fs);seen(key)=char(source);
        rows(end+1,:)={destination,label,gender,disease,subject,vowel,source,fs,numel(x)}; %#ok<AGROW>
        logs(end+1,:)={i,source,destination,"Completed","Native sample rate; exact PCM16 payload"}; %#ok<AGROW>
    catch err
        logs(end+1,:)={i,source,destination,"Failed",string(err.message)}; %#ok<AGROW>
    end
end
log=cell2table(logs,'VariableNames',{'CSVRow','SourceNSP','OutputWAV','Status','Detail'});
writetable(log,fullfile(outputWavFolder,'NSP_Conversion_QC.xlsx'));
assert(~isempty(rows),'No recordings converted. Inspect NSP_Conversion_QC.xlsx.');
T=cell2table(rows,'VariableNames',{'FilePath','Label','Gender','Disease','SubjectID','Vowel','SourceNSP','NativeSampleRate','OriginalSampleCount'});
file=fullfile(outputWavFolder,'Thesis_Final_Dataset.xlsx');tmp=[tempname(outputWavFolder) '.xlsx'];
writetable(T,tmp);movefile(tmp,file,'f');
fprintf('Converted/verified %d recordings. Metadata: %s\n',height(T),file);
fprintf('Original WAVs/features are unchanged. Review every Failed/ExcludedDisease row before extracting new features.\n');
end

function [x,fs]=readNSP16(file)
% Parse the actual FORMDS16 container and the HEDR/SDA_ chunks.
fid=fopen(file,'rb','ieee-le');assert(fid>=0,'Cannot open %s.',file);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fseek(fid,0,'eof');bytes=ftell(fid);fseek(fid,0,'bof');
magic=char(fread(fid,8,'uint8=>char')');assert(strcmp(magic,'FORMDS16'),'Unsupported NSP container; expected FORMDS16.');
size=fread(fid,1,'uint32=>double');assert(isscalar(size)&&size==bytes-12,'Container length mismatch.');
fs=[];expected=[];x=[];dataSeen=false;headerSeen=false;
while ftell(fid)<bytes
    assert(bytes-ftell(fid)>=8,'Truncated NSP chunk header.');
    tag=char(fread(fid,4,'uint8=>char')');count=fread(fid,1,'uint32=>double');start=ftell(fid);
    assert(isscalar(count)&&start+count<=bytes,'Truncated NSP chunk payload.');
    switch tag
        case 'HEDR'
            assert(~headerSeen&&count>=28,'Invalid or repeated HEDR chunk.');headerSeen=true;
            fseek(fid,start+20,'bof');fs=fread(fid,1,'uint32=>double');expected=fread(fid,1,'uint32=>double');
        case 'SDA_'
            assert(~dataSeen&&mod(count,2)==0,'Repeated or malformed SDA_ chunk.');dataSeen=true;
            x=fread(fid,count/2,'int16=>int16');
        case {'SDB_','SDAB'}
            error('Multiple-channel NSP is not supported; no channel was silently discarded.');
    end
    assert(fseek(fid,start+count,'bof')==0,'Cannot seek to next NSP chunk.');
end
assert(headerSeen&&dataSeen&&~isempty(x),'Missing audio/header chunk.');
assert(isscalar(fs)&&fs>=1000&&fs<=384000,'Invalid native sample rate.');
assert(numel(x)==expected,'Header sample count differs from audio payload.');
end

function writeNativeWAV(file,x,fs)
folder=fileparts(file);if isempty(folder),folder=pwd;end;if ~isfolder(folder),mkdir(folder);end
expected=double(x(:))/32768;
if isfile(file)
    [old,oldFs]=audioread(file);
    assert(oldFs==fs&&isequal(old,expected),'Existing output differs; refusing to overwrite it: %s',file);
    return;
end
tmp=[tempname(folder) '.wav'];audiowrite(tmp,expected,fs,'BitsPerSample',16);
[y,newFs]=audioread(tmp);assert(newFs==fs&&isequal(y,expected),'WAV did not preserve exact PCM samples.');
movefile(tmp,file,'f');
end

function key=nspNameKey(name)
key=lower(regexprep(char(name),'[\s_-]',''));
end

function path=resolveNSP(raw,index)
raw=char(raw);[~,name,ext]=fileparts(raw);
if strcmpi(ext,'.nsp')&&isfile(raw),path=raw;return;end
key=nspNameKey(name);assert(isKey(index,key),'No source NSP matches the CSV basename.');
q=index(key);assert(numel(q)==1,'Ambiguous NSP basename: %d sources found. Use an exact source path in CSV.',numel(q));path=q{1};
end

function disease=detectNSPDisease(path)
% Match complete folder components, or an explicit disease filename prefix.
% Never infer a patient's diagnosis from a substring such as Normal in Abnormal.
alias=["Healthy","Normal","Hyperfunktionelle","Hyperfunktionelle Dysphonie", ...
    "Hypofunktionelle","Hypofunktionelle Dysphonie","Laryngitis","Rekurrensparese", ...
    "Reinke Ödem","Reinke","Spasmodische Dysphonie","Stimmlippenpolyp"];
canonical=["Healthy","Healthy","Hyperfunktionelle","Hyperfunktionelle", ...
    "Hypofunktionelle","Hypofunktionelle","Laryngitis","Rekurrensparese", ...
    "Reinke Ödem","Reinke Ödem","Spasmodische Dysphonie","Stimmlippenpolyp"];
pieces=string(regexp(strrep(char(path),'\','/'),'/','split'));found=strings(0,1);
for j=1:numel(alias)
    equal=strcmpi(strip(pieces),alias(j));prefix=startsWith(pieces(end),alias(j)+"_",'IgnoreCase',true);
    if any(equal)||prefix,found(end+1,1)=canonical(j);end %#ok<AGROW>
end
found=unique(found);assert(numel(found)<=1,'Multiple disease folders/prefixes in source path.');
if isempty(found),disease="";else,disease=found(1);end
end

function nspSelfTest
folder=tempname;mkdir(folder);file=fullfile(folder,'106-a_n.nsp');
x=int16([-32768;-20000;-1;0;1;20000;32767]);fs=50000;writeSyntheticNSP(file,x,fs,numel(x));
[y,rate]=readNSP16(file);assert(rate==fs&&isequal(x,y));
wav=fullfile(folder,'106-a_n.wav');writeNativeWAV(wav,y,rate);[r,rate]=audioread(wav);
assert(rate==fs&&isequal(r,double(x)/32768));writeNativeWAV(wav,y,rate);
bad=fullfile(folder,'bad.nsp');writeSyntheticNSP(bad,x,fs,numel(x)+1);
failed=false;try,readNSP16(bad);catch,failed=true;end;assert(failed,'Malformed sample count was accepted.');
index=containers.Map('KeyType','char','ValueType','any');index(nspNameKey('106-a_n'))={file,bad};
failed=false;try,resolveNSP('unknown/106-a_n.nsp',index);catch,failed=true;end;assert(failed,'Ambiguous basename was accepted.');
assert(isempty(detectNSPDisease('D:/Abnormal/106-a_n.nsp')));
assert(detectNSPDisease('D:/Hyperfunktionelle/106-a_n.nsp')=="Hyperfunktionelle");
fprintf('NSP_PARSER_EXACT_PCM_WAV_AND_AMBIGUITY_CHECKS_PASSED\n');
end

function writeSyntheticNSP(file,x,fs,declaredCount)
fid=fopen(file,'wb','ieee-le');assert(fid>=0);cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,'FORMDS16','char');fwrite(fid,48+2*numel(x),'uint32');
fwrite(fid,'HEDR','char');fwrite(fid,32,'uint32');fwrite(fid,'Jan 01 00:00:00 2000','char');
fwrite(fid,fs,'uint32');fwrite(fid,declaredCount,'uint32');fwrite(fid,[32767,-1],'int16');
fwrite(fid,'SDA_','char');fwrite(fid,2*numel(x),'uint32');fwrite(fid,x,'int16');
end

function compare_test_forward_models(inputDir, outputDir)
% Run the original FWD3D induced-magnetic TMI operator on exported E01 models.
% Outputs are one resumable MAT file per sample in X-fastest receiver order.

if nargin < 1 || isempty(inputDir)
    inputDir = fullfile('analysis_outputs','matlab_python_tmi','inputs');
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile('analysis_outputs','matlab_python_tmi','matlab_outputs');
end
repoRoot = fileparts(fileparts(mfilename('fullpath')));
inputDir = absolute_path(repoRoot,inputDir);
outputDir = absolute_path(repoRoot,outputDir);
if ~exist(outputDir,'dir'); mkdir(outputDir); end
addpath(fullfile(repoRoot,'matlab_code','FWD3D','bin'));

% Exact E01 grid, receiver geometry, field, and coordinate conventions.
gPars = getGpars([0 640 0 640 0 240],[10 10],10);
[xq,yq,zq,w] = getQuadPoints(gPars.xg,gPars.yg,gPars.zg,gPars.dx,gPars.dy,gPars.dzg,1);
[rxGrid,ryGrid] = ndgrid(0:10:800,0:10:800);
rx=rxGrid(:); ry=ryGrid(:); rz=-10*ones(size(rx)); rc=1;
Bo=50000; Ao=0; Io=75; Do=25;

files=dir(fullfile(inputDir,'sample_*.h5'));
if isempty(files); error('No sample HDF5 files found in %s',inputDir); end
for index=1:numel(files)
    source=fullfile(files(index).folder,files(index).name);
    destination=fullfile(outputDir,files(index).name);
    if exist(destination,'file')
        fprintf('Skipping existing %d/%d: %s\n',index,numel(files),files(index).name);
        continue
    end
    susceptibility_xfast_si=h5read(source,'/susceptibility_xfast_si');
    susceptibility_xfast_si=susceptibility_xfast_si(:);
    if numel(susceptibility_xfast_si) ~= 24*64*64
        error('%s has an unexpected susceptibility length',files(index).name);
    end
    matlab_tmi_xfast_nt=getPredMag(xq,yq,zq,w,susceptibility_xfast_si,rx,ry,rz,rc,Bo,Ao,Io,Do);
    if exist(destination,'file'); delete(destination); end
    h5create(destination,'/matlab_tmi_xfast_nt',size(matlab_tmi_xfast_nt));
    h5write(destination,'/matlab_tmi_xfast_nt',matlab_tmi_xfast_nt);
    fprintf('MATLAB forward %d/%d: %s\n',index,numel(files),files(index).name);
end
fprintf('MATLAB outputs written to %s\n',outputDir);
end

function result=absolute_path(repoRoot,value)
if ispc
    isAbsolute=~isempty(regexp(value,'^[A-Za-z]:[\\/]','once'));
else
    isAbsolute=startsWith(value,filesep);
end
if isAbsolute; result=value; else; result=fullfile(repoRoot,value); end
end

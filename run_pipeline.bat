@echo off
REM run_pipeline.bat - the gold pipeline in order (Windows, conda environment from environment.yml).
REM Data are not in this repository: see README.md, "Data". Steps marked (internet) fetch open data.
REM     run_pipeline.bat      > pipeline.log 2>&1    inputs, V1 model, targets, shortlist, desk check
REM     run_pipeline.bat gsi  > pipeline_gsi.log 2>&1  after downloading GSI reports into Reports\<report_id>\
setlocal
set RAW=data\raw\aikosh-gsi-ka-ap
set PROC=data\processed
set PF=gc_au_index,gc_au,gc_as,gc_sb,gc_bi
set V2=%PF%,struct_,v2s_,v2g_
if /i "%1"=="gsi" goto :gsi

echo === 1. labels and input layers ===
python prepare_vectors.py %RAW% %PROC%\ka_ap_v1.gpkg || goto :fail
python build_raster_stack.py %RAW% %PROC% || goto :fail
python build_vector_features.py %RAW% %PROC% || goto :fail
python build_geochem_grid.py %PROC% || goto :fail
python build_terrain_layers.py %PROC% || goto :fail
REM (internet) NGDR map-service layers used by V2
python ngdr_wfs_pull.py %PROC% --max-layers 200 || goto :fail
echo === 2. V1 model: five geochemistry layers, monotonic, 8 bags ===
python train_v1.py %PROC% --name pf_ens --only %PF% --monotonic --bags 8 || goto :fail
python make_targets.py %PROC% --model pf_ens --stability bags || goto :fail
python shortlist_targets.py %PROC% --model pf_ens || goto :fail
REM (internet) Sentinel-2 imagery and OpenStreetMap
python auto_desk_check.py %PROC% %RAW% --model pf_ens || goto :fail
python gsi_reports.py fetchlist %PROC% --model pf_ens || goto :fail
echo === done: download the listed GSI reports, then run:  run_pipeline.bat gsi ===
exit /b 0

:gsi
echo === 3. GSI exploration results: graded sites, blind test ===
python gsi_labels.py %PROC% --reports-dir Reports --model pf_ens || goto :fail
python gsi_eval.py %PROC% --models pf_ens --blend none || goto :fail
python v1_summary.py %PROC% --model pf_ens || goto :fail
echo === 4. V2 contenders and verdict ===
python build_v2_layers.py %PROC% || goto :fail
python train_v1.py %PROC% --name v2_struct --only %PF%,struct_,v2s_ --monotonic --bags 8 || goto :fail
python train_v1.py %PROC% --name v2_all --only %V2% --monotonic --bags 8 || goto :fail
python train_v1.py %PROC% --name v2_b35 --only %V2% --monotonic --bags 8 --belt-km 3.5 || goto :fail
python train_v1.py %PROC% --name cnn_pf --only %PF% --learner cnn --bags 8 || echo === CNN skipped: PyTorch did not load ===
python train_v1.py %PROC% --name cnn_all --only %V2% --learner cnn --bags 8 || echo === CNN skipped: PyTorch did not load ===
python compare_runs.py %PROC% pf_ens cnn_pf v2_struct v2_all cnn_all v2_b35 || goto :fail
python gsi_eval.py %PROC% --models pf_ens cnn_pf v2_struct v2_all cnn_all --blend none --tag _v2 || goto :fail
python gsi_eval.py %PROC% --models pf_ens v2_all v2_b35 --blend none --belt-km 3.5 --tag _v2_belt3.5km || goto :fail
python v2_verdict.py %PROC% || goto :fail
echo === done ===
exit /b 0

:fail
echo === STOPPED: the step above failed ===
exit /b 1
